import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import torch.distributed as dist


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Dict[str, Any]) -> None:
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_git_commit_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"


def get_env_info() -> Dict[str, str]:
    info = {
        "python_version": sys.version.replace("\n", " "),
        "torch_version": torch.__version__,
        "cuda_available": str(torch.cuda.is_available()),
        "cuda_version": str(torch.version.cuda),
        "cudnn_version": str(torch.backends.cudnn.version()),
    }
    try:
        import deepspeed  # type: ignore

        info["deepspeed_version"] = deepspeed.__version__
    except Exception:
        info["deepspeed_version"] = "not_installed"
    return info


def get_rank_world_size_local_rank() -> Dict[str, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return {"rank": rank, "world_size": world_size, "local_rank": local_rank}


def init_distributed_if_needed() -> bool:
    env = get_rank_world_size_local_rank()
    if env["world_size"] <= 1:
        return False
    if dist.is_available() and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    return dist.is_initialized()


def barrier_if_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def is_main_process() -> bool:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True

