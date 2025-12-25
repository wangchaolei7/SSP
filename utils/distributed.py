import os
import shutil
import random
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import math


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_distributed() else 0


def get_world_size():
    return dist.get_world_size() if is_distributed() else 1


def get_local_rank():
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_main_process():
    return get_rank() == 0


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(get_local_rank())
        return True
    return False


def cleanup_distributed():
    if is_distributed():
        dist.destroy_process_group()


def barrier():
    if is_distributed():
        dist.barrier()


def all_reduce_tensor(tensor, op=dist.ReduceOp.SUM):
    if not is_distributed():
        return tensor
    if not tensor.is_cuda:
        tensor = tensor.to(torch.device("cuda"))
    dist.all_reduce(tensor, op=op)
    return tensor


def all_reduce_scalar(value, op="mean"):
    if not is_distributed():
        return value
    tensor = torch.tensor(value, device=torch.device("cuda"), dtype=torch.float32)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    if op == "mean":
        tensor = tensor / get_world_size()
    return tensor.item()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def merge_rank_outputs(base_dir, rank_dirs, cleanup=True):
    for rank_dir in rank_dirs:
        if not os.path.isdir(rank_dir):
            continue
        for root, _, files in os.walk(rank_dir):
            rel = os.path.relpath(root, rank_dir)
            target_root = base_dir if rel == "." else os.path.join(base_dir, rel)
            os.makedirs(target_root, exist_ok=True)
            for name in files:
                src = os.path.join(root, name)
                dst = os.path.join(target_root, name)
                if not os.path.exists(dst):
                    shutil.move(src, dst)
        if cleanup:
            shutil.rmtree(rank_dir, ignore_errors=True)


class DistributedEvalSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False, seed=0, drop_last=False):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed, drop_last=drop_last)

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(indices), generator=g).tolist()
        indices = indices[self.rank::self.num_replicas]
        return iter(indices)

    def __len__(self):
        if len(self.dataset) <= self.rank:
            return 0
        return math.ceil((len(self.dataset) - self.rank) / self.num_replicas)
