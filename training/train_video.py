import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import PIL.Image as Image
    import os
    import numpy as np
    import random
    from tqdm import tqdm
    import datetime
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    import time
    import yaml

    from data.dataset_prep import prep_video_dataset
    from utils.optim_utils import get_criterion, get_optimizer_scheduler
    from training.utils.utils_video import save_model

    from models.image.models import get_model as get_image_model
    from models.opt_flow import get_flow_model
    from models.video.models_consistency import get_model as get_video_model
    import models.video.models_consistency as models_consistency
    from utils.distributed import (
        cleanup_distributed,
        is_main_process,
        all_reduce_scalar,
        all_reduce_tensor,
        seed_everything,
        unwrap_model,
    )

import argparse
def init_distributed_and_device():
    import os
    import torch
    import torch.distributed as dist

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = dist.is_available() and world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if distributed and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")

    return distributed, rank, local_rank, world_size, device

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Training Parameters")
    parser.add_argument("config", metavar="C", type=str, help="Name of config file")
    parser.add_argument("--gpus", required=False, type=str, default=None,
                        help="Comma-separated GPU ids to use, e.g. \"0,1,3\"")
    return parser.parse_args()

def _resolve_config_path(config, base_dir):
    if os.path.isfile(config):
        return config
    return os.path.join(base_dir, config)

def normalize_state_dict_keys(state_dict, model):
    if not state_dict:
        return state_dict
    model_state = unwrap_model(model).state_dict()
    if not model_state:
        return state_dict
    state_has_module = any(k.startswith("module.") for k in state_dict.keys())
    model_has_module = any(k.startswith("module.") for k in model_state.keys())
    if state_has_module and not model_has_module:
        return {k[len("module."):] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    if not state_has_module and model_has_module:
        return {f"module.{k}" if not k.startswith("module.") else k: v for k, v in state_dict.items()}
    return state_dict

def resolve_checkpoint_path(save_dir, checkpoint_name):
    run_dir = os.path.join(save_dir, checkpoint_name)
    candidates = [
        os.path.join(run_dir, "latest.pth.tar"),
        os.path.join(run_dir, f"{checkpoint_name.split('@')[-1]}.pth.tar"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def main(config):
    distributed, rank, local_rank, world_size, device = init_distributed_and_device()
    date = datetime.datetime.now()
    def _tqdm(*args, **kwargs):
        kwargs.setdefault("disable", not is_main_process())
        return tqdm(*args, **kwargs)
    models_consistency.tqdm = _tqdm
    # Config file
    config_path = _resolve_config_path(config, "config/video")
    with open(config_path, 'r') as cfg_file:
        cfg = yaml.load(cfg_file, Loader=yaml.FullLoader)
    
    resume_training = cfg["resume_training"] is not None
    if resume_training:
        checkpoint_name = cfg["resume_training"]
        config = os.path.join(cfg["save_dir"], checkpoint_name, checkpoint_name.split("@")[-1] + "_config.yaml")
        with open(config, 'r') as cfg_file:
            cfg = yaml.load(cfg_file, Loader=yaml.FullLoader)
        date = datetime.datetime.strptime(checkpoint_name.split("@")[-1], "%d-%m_%H-%M") 
        resume_path = resolve_checkpoint_path(cfg["save_dir"], checkpoint_name)
        if resume_path is None:
            raise FileNotFoundError(f"No checkpoint found for {checkpoint_name} in {cfg['save_dir']}")
        resume_checkpoint = torch.load(resume_path, map_location="cpu")
        if is_main_process():
            print(date)

    data_cfg = cfg["data_cfg"]
    image_model_cfg = cfg["image_model_cfg"]
    video_model_cfg = cfg["video_model_cfg"]
    training_cfg = cfg["training_cfg"]
    optim_cfg = cfg["optim_cfg"]
    loss_cfg = cfg["loss_cfg"]
    save_every = training_cfg.get("save_every", 0)
    save_every = int(save_every) if save_every else 0
    save_last = training_cfg.get("save_last", True)

    if is_main_process():
        visible_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
        print(f"Distributed: {distributed} | world_size={world_size} | rank={rank} | local_rank={local_rank} | cuda_device={torch.cuda.current_device()} | GPUs={visible_gpus}")

    base_seed = cfg.get("seed", 1337)
    seed_everything(base_seed + rank)

    def worker_init_fn(worker_id):
        seed = base_seed + rank + worker_id
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Dataset
    train_dataset, val_dataset, DATASET = prep_video_dataset(data_cfg)

    # Dataloader
    batch_size = training_cfg["batch_size"]
    train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True) if distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=True) if distributed else None
    persistent_workers = training_cfg["num_workers"] > 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=training_cfg["num_workers"],
        persistent_workers=persistent_workers,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        drop_last=True,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=training_cfg["num_workers"],
        persistent_workers=persistent_workers,
        sampler=val_sampler,
        shuffle=False,
        drop_last=True,
        worker_init_fn=worker_init_fn,
    )

    # Trained image model
    image_save_dir = image_model_cfg["image_save_dir"]
    img_checkpoint_folder = image_model_cfg["checkpoint_folder"]
    img_checkpoint_name = image_model_cfg["checkpoint_name"]
    img_best_model = image_model_cfg["best_model"]
    with open(os.path.join(image_save_dir, img_checkpoint_folder + img_checkpoint_name, img_checkpoint_name.split("@")[-1] + "_config.yaml"), 'r') as cfg_file:
        image_cfg = yaml.load(cfg_file, Loader=yaml.FullLoader)
    seg_model = get_image_model(image_cfg["model_cfg"], DATASET.n_classes)
    seg_model.to(device)
    if img_best_model:
        img_checkpoint = torch.load(os.path.join(image_cfg["save_dir"], img_checkpoint_folder + img_checkpoint_name, "best_model_" + img_checkpoint_name.split("@")[-1] + ".pth.tar"), map_location="cpu")
    else:
        img_checkpoint = torch.load(os.path.join(image_cfg["save_dir"], img_checkpoint_folder + img_checkpoint_name, img_checkpoint_name.split("@")[-1] + ".pth.tar"), map_location="cpu")
    seg_model.load_state_dict(img_checkpoint["model"])

    # Video model
    flow_model = get_flow_model()
    flow_model.to(device)

    model = get_video_model(video_model_cfg, seg_model, flow_model, DATASET.n_classes)
    model.to(device)
    if is_main_process():
        print(f"Model has {sum([p.numel() for p in model.parameters()]):,} parameters")

    # Load pre-trained checkpoint
    if cfg["pretrained_checkpoint"] is not None:
        checkpoint_name = cfg["pretrained_checkpoint"]
        checkpoint_path = resolve_checkpoint_path(cfg["save_dir"], checkpoint_name)
        if checkpoint_path is not None:
            pretrained_checkpoint = torch.load(checkpoint_path, map_location="cpu")
            state_dict = pretrained_checkpoint["model"]
        else:
            state_dict = torch.load(checkpoint_name)
        state_dict = normalize_state_dict_keys(state_dict, model)
        unwrap_model(model).load_state_dict(state_dict, strict=False)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )
    core_model = unwrap_model(model)

    # Optim
    iter_per_epoch = len(train_loader)
    criterion = get_criterion(loss_cfg, DATASET.ignore_index, device=device, soft_labels=data_cfg.get("soft_labels", False))
    if training_cfg["trained_layer"] is not None:
        param_groups = [{'params': (p for (n, p) in model.named_parameters() if (training_cfg["trained_layer"] not in n)), 'lr': training_cfg["secondary_lr"]}, 
                        {'params': (p for (n, p) in model.named_parameters() if (training_cfg["trained_layer"] in n)), 'lr': training_cfg["lr"]}]
    else:
        param_groups = model.parameters()
    
    optimizer, scheduler = get_optimizer_scheduler(optim_cfg, param_groups, iter_per_epoch, lr=training_cfg["lr"], num_epochs=training_cfg["num_epochs"])
    if is_main_process():
        print(f"{iter_per_epoch} iterations per epochs of {batch_size} batches each")

    start_epoch = 0
    train_losses = []
    train_const_losses = []
    val_losses = []
    val_const_losses = []
    val_global_miou = []
    val_miou1 = []
    val_miou2 = []
    val_classes_mIoU = []

    if resume_training:
        start_epoch = resume_checkpoint["epoch"]
        resume_state = normalize_state_dict_keys(resume_checkpoint["model"], model)
        core_model.load_state_dict(resume_state)
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        scheduler.load_state_dict(resume_checkpoint["scheduler"])

        train_losses = resume_checkpoint["train_losses"]
        train_const_losses = resume_checkpoint["train_const_losses"]
        val_losses = resume_checkpoint["val_losses"]
        val_const_losses = resume_checkpoint["val_const_losses"]
        val_global_miou = resume_checkpoint["val_global_miou"]
        val_miou1 = resume_checkpoint["val_miou1"]
        val_miou2 = resume_checkpoint["val_miou2"]
        val_classes_mIoU = resume_checkpoint["val_classes_mIoU"]

    epochs = training_cfg["early_stopping"] if training_cfg["early_stopping"] is not None else training_cfg["num_epochs"]
    for epoch in range(start_epoch, epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if data_cfg.get("logit_distillation", False):
            train_loss, const_loss = core_model.kd_train_one_epoch(
            train_loader, 
            optimizer, 
            criterion, 
            scheduler, 
            device,
            )
        else:
            train_loss, const_loss = core_model.train_one_epoch(
                train_loader, 
                optimizer, 
                criterion, 
                scheduler, 
                device,
                )
        train_loss = all_reduce_scalar(train_loss, op="mean")
        const_loss = all_reduce_scalar(const_loss, op="mean")
        train_losses.append(train_loss)
        train_const_losses.append(const_loss)

        results = core_model.evaluate_with_metrics(
            val_loader, 
            criterion, 
            device,
            n_classes=DATASET.n_classes,
            ignore_index=DATASET.ignore_index,
            )
        results["val_loss"] = all_reduce_scalar(results["val_loss"], op="mean")
        results["const_loss"] = all_reduce_scalar(results["const_loss"], op="mean")
        if torch.is_tensor(results["global_miou"]):
            results["global_miou"] = all_reduce_tensor(results["global_miou"]).div_(world_size).cpu()
        if torch.is_tensor(results["classes_mIoU"]):
            results["classes_mIoU"] = all_reduce_tensor(results["classes_mIoU"]).div_(world_size).cpu()
        val_losses.append(results['val_loss'])
        val_const_losses.append(results['const_loss'])
        val_global_miou.append(results["global_miou"])
        val_miou1.append(results["mIoU1"])
        val_miou2.append(results["mIoU2"])
        val_classes_mIoU.append(results["classes_mIoU"])
        if is_main_process():
            per_classes_mIoU = {v: results["classes_mIoU"][k-1].item() for (k,v) in DATASET.classes.items() if k>0} if DATASET.ignore_index > 0 else {v: results["classes_mIoU"][k].item() for (k,v) in DATASET.classes.items()}
            print(f"Epoch {epoch+1}: train loss = {train_loss:.4f}, train consist loss = {const_loss:.4f}, validation loss = {val_losses[-1]:.4f}, val consist loss = {val_const_losses[-1]:.4f}, val mIoU = {val_global_miou[-1]:.4f}")
            print("-"*70)
        
        save_name = None
        if is_main_process():
            save_dict = {
                "model": core_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch+1,
                "train_losses": train_losses,
                "train_const_losses": train_const_losses,
                "val_losses": val_losses,
                "val_const_losses": val_const_losses,
                "val_global_miou": val_global_miou,
                "val_miou1": val_miou1,
                "val_miou2": val_miou2,
                "val_classes_mIoU": val_classes_mIoU
            }
            save_name = save_model(
                save_dict,
                cfg,
                cfg["save_dir"],
                date,
                per_classes_mIoU=per_classes_mIoU,
                checkpoint_name="latest.pth.tar",
                save_checkpoint=save_last,
            )
            if save_every > 0 and (epoch + 1) % save_every == 0:
                run_dir = os.path.join(cfg["save_dir"], date.strftime("%d-%m_%H-%M"))
                milestone_name = f"epoch_{epoch+1:04d}.pth.tar"
                torch.save(save_dict, os.path.join(run_dir, milestone_name))

    return save_name


if __name__=="__main__":
    args = parse_args()
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    config = args.config
    save_name = main(config)
    if is_main_process():
        print(save_name)
    cleanup_distributed()
