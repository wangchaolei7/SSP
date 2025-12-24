import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import os
    import numpy as np
    import random
    import torch
    import yaml
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    import datetime

    from data.dataset_prep import prep_image_dataset
    from utils.optim_utils import get_criterion, get_optimizer_scheduler
    from models.image.models import get_model
    from training.utils.utils_image import train_one_epoch, train_with_metrics, evaluate, evaluate_with_metrics, save_model
    from utils.distributed import (
        setup_distributed,
        cleanup_distributed,
        is_main_process,
        get_rank,
        get_world_size,
        get_local_rank,
        all_reduce_scalar,
        all_reduce_tensor,
        seed_everything,
        barrier,
    )

import argparse
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


def main(config):
    distributed = setup_distributed()
    rank = get_rank()
    world_size = get_world_size()
    local_rank = get_local_rank()
    date = datetime.datetime.now()
    # Config file
    config_path = _resolve_config_path(config, "config/image")
    with open(config_path, 'r') as cfg_file:
        cfg = yaml.load(cfg_file, Loader=yaml.FullLoader)

    save_dir = cfg["save_dir"]
    resume_training = cfg["resume_training"] is not None
    if resume_training:
        checkpoint_name = cfg["resume_training"]
        config = os.path.join(save_dir, checkpoint_name, checkpoint_name.split("@")[-1] + "_config.yaml")
        with open(config, 'r') as cfg_file:
            cfg = yaml.load(cfg_file, Loader=yaml.FullLoader)
        date = datetime.datetime.strptime(checkpoint_name.split("@")[-1], "%d-%m_%H-%M") 
        resume_checkpoint = torch.load(os.path.join(save_dir, checkpoint_name, checkpoint_name.split("@")[-1] + ".pth.tar"), map_location="cpu")

    data_cfg = cfg["data_cfg"]
    model_cfg = cfg["model_cfg"]
    training_cfg = cfg["training_cfg"]
    optim_cfg = cfg["optim_cfg"]
    loss_cfg = cfg["loss_cfg"]

    if is_main_process():
        visible_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
        print(f"Distributed: {distributed} | world_size={world_size} | rank={rank} | local_rank={local_rank} | GPUs={visible_gpus}")

    base_seed = cfg.get("seed", 1337)
    seed_everything(base_seed + rank)

    def worker_init_fn(worker_id):
        seed = base_seed + rank + worker_id
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Dataset
    train_dataset, val_dataset, DATASET = prep_image_dataset(data_cfg)

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

    # Model
    device = torch.device("cuda", local_rank)
    model = get_model(
        model_cfg,
        DATASET.n_classes,
        )
    model.to(device)

    # Load pre-trained checkpoint
    if cfg["pretrained_checkpoint"] is not None:
        checkpoint = torch.load(os.path.join(cfg["save_dir"], cfg["pretrained_checkpoint"], "best_model_" + cfg["pretrained_checkpoint"].split("@")[-1] + ".pth.tar"), map_location="cpu")
        state_dict = checkpoint["model"]
        state_dict = {k: v for (k, v) in state_dict.items() if "classifier" not in k}
        model.load_state_dict(state_dict, strict=False)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
        )

    # Optim
    iter_per_epoch = len(train_loader)
    criterion = get_criterion(loss_cfg, DATASET.ignore_index, device=device, soft_labels=data_cfg.get("soft_labels", False))
    optimizer, scheduler = get_optimizer_scheduler(optim_cfg, model.parameters(), iter_per_epoch, lr=training_cfg["lr"], num_epochs=training_cfg["num_epochs"])
    if is_main_process():
        print(f"{iter_per_epoch} iterations per epochs of {batch_size} batches each")

    start_epoch = 0
    train_losses = []
    train_global_miou = []
    train_classes_mIoU = []
    val_losses = []
    val_global_miou = []
    val_miou1 = []
    val_miou2 = []
    val_classes_mIoU = []

    if cfg.get("no_training", False):
        save_dict = {
            "model": model.state_dict() if not distributed else model.module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": 0,
            "train_losses": [torch.tensor(0)],
            "train_global_miou": [torch.tensor(0)],
            "train_classes_mIoU": [[torch.tensor(0) for _ in range(DATASET.n_classes)]],
            "val_losses": [torch.tensor(0)],
            "val_global_miou": [torch.tensor(0)],
            "val_miou1": [torch.tensor(0)],
            "val_miou2": [torch.tensor(0)],
            "val_classes_mIoU": [[torch.tensor(0) for _ in range(DATASET.n_classes)]]
        }
        save_name = None
        if is_main_process():
            save_name = save_model(save_dict, cfg, cfg["save_dir"], date, DATASET)
        barrier()
        return save_name


    if resume_training:
        if distributed:
            model.module.load_state_dict(resume_checkpoint["model"])
        else:
            model.load_state_dict(resume_checkpoint["model"])
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        scheduler.load_state_dict(resume_checkpoint["scheduler"])
        start_epoch = resume_checkpoint["epoch"]

        train_losses = resume_checkpoint["train_losses"]
        train_global_miou = resume_checkpoint["train_global_miou"]
        train_classes_mIoU = resume_checkpoint["train_classes_mIoU"]
        val_losses = resume_checkpoint["val_losses"]
        val_global_miou = resume_checkpoint["val_global_miou"]
        val_miou1 = resume_checkpoint["val_miou1"]
        val_miou2 = resume_checkpoint["val_miou2"]
        val_classes_mIoU = resume_checkpoint["val_classes_mIoU"]

    epochs = training_cfg["early_stopping"] if training_cfg["early_stopping"] is not None else training_cfg["num_epochs"]
    for epoch in range(start_epoch, epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_loss = train_one_epoch(
            train_loader, 
            model,
            optimizer, 
            criterion, 
            scheduler, 
            device,
            disable_tqdm=not is_main_process(),
            )
        train_loss = all_reduce_scalar(train_loss, op="mean")
        train_global_miou_ep = torch.tensor(0)
        train_classes_mIoU_ep = [torch.tensor(0) for _ in range(DATASET.n_classes)]
        # Compute metrics during training:
        #train_loss, train_global_miou_ep, train_classes_mIoU_ep = train_with_metrics(
        #    train_loader, 
        #    model,
        #    optimizer, 
        #    criterion, 
        #    scheduler, 
        #    device,
        #    DATASET.n_classes,
        #    DATASET.ignore_index
        #    )
        train_losses.append(train_loss)
        train_global_miou.append(train_global_miou_ep)
        train_classes_mIoU.append(train_classes_mIoU_ep)

        results = evaluate_with_metrics(
            val_loader, 
            model, 
            criterion, 
            device,
            n_classes=DATASET.n_classes,
            ignore_index=DATASET.ignore_index,
            disable_tqdm=not is_main_process(),
            )
        val_loss = all_reduce_scalar(results["val_loss"], op="mean")
        results["val_loss"] = val_loss
        if torch.is_tensor(results["global_miou"]):
            results["global_miou"] = all_reduce_tensor(results["global_miou"]).div_(world_size).cpu()
        if torch.is_tensor(results["classes_mIoU"]):
            results["classes_mIoU"] = all_reduce_tensor(results["classes_mIoU"]).div_(world_size).cpu()
        val_losses.append(results['val_loss'])
        val_global_miou.append(results["global_miou"])
        val_miou1.append(results["mIoU1"])
        val_miou2.append(results["mIoU2"])
        val_classes_mIoU.append(results["classes_mIoU"])
        if is_main_process():
            per_classes_mIoU = {v: results["classes_mIoU"][k-1].item() for (k,v) in DATASET.classes.items() if k>0} if DATASET.ignore_index > 0 else {v: results["classes_mIoU"][k].item() for (k,v) in DATASET.classes.items()}
            print(f"Epoch {epoch+1}: train loss = {train_loss:.4f}, validation loss = {val_losses[-1]:.4f}, val mIoU = {val_global_miou[-1]:.4f}")
            print("-"*70)
        
        save_name = None
        if is_main_process():
            save_dict = {
                "model": model.state_dict() if not distributed else model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch+1,
                "train_losses": train_losses,
                "train_global_miou": train_global_miou,
                "train_classes_mIoU": train_classes_mIoU,
                "val_losses": val_losses,
                "val_global_miou": val_global_miou,
                "val_miou1": val_miou1,
                "val_miou2": val_miou2,
                "val_classes_mIoU": val_classes_mIoU
            }
            save_name = save_model(save_dict, cfg, cfg["save_dir"], date, DATASET)

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
