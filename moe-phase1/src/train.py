import argparse
import json
import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from synthetic_data import build_dataloader
from utils import ensure_dir, get_env_info, get_git_commit_hash, load_json, save_json, set_seed


class ExpertMLP(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, 4 * hidden_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4 * hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class SimpleMoE(nn.Module):

    def __init__(self, hidden_size: int, num_experts: int, top_k: int) -> None:
        super().__init__()
        self.num_experts = int(num_experts)
        self.top_k = max(1, min(int(top_k), self.num_experts))
        self.gate = nn.Linear(hidden_size, self.num_experts)
        self.experts = nn.ModuleList([ExpertMLP(hidden_size) for _ in range(self.num_experts)])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, hidden = x.shape
        flat = x.reshape(-1, hidden)  

        gate_logits = self.gate(flat)  
        topk_vals, topk_idx = torch.topk(gate_logits, k=self.top_k, dim=-1)
        topk_w = torch.softmax(topk_vals, dim=-1)

        out_flat = torch.zeros_like(flat)

        for expert_id, expert in enumerate(self.experts):
            mask = (topk_idx == expert_id)
            token_pos, slot = mask.nonzero(as_tuple=True)
            if token_pos.numel() == 0:
                continue

            expert_in = flat[token_pos]
            expert_out = expert(expert_in)
            w = topk_w[token_pos, slot].unsqueeze(-1)
            out_flat[token_pos] += expert_out * w

        out = out_flat.view(bsz, seq_len, hidden)

        probs = torch.softmax(gate_logits, dim=-1).mean(dim=0)
        l_aux = (probs * probs).sum() * self.num_experts

        return out, l_aux


class MoEBlock(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int, top_k: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.moe = SimpleMoE(hidden_size, num_experts, top_k)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y, l_aux = self.moe(self.norm(x))
        return x + y, l_aux


class TinyMoELM(nn.Module):
    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.vocab_size = int(config["vocab_size"])
        self.hidden_size = int(config["hidden_size"])
        self.num_layers = int(config["num_layers"])
        self.num_experts = int(config["num_experts"])
        self.top_k = int(config["top_k"])

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.blocks = nn.ModuleList(
            [MoEBlock(self.hidden_size, self.num_experts, self.top_k) for _ in range(self.num_layers)]
        )
        self.final_norm = nn.LayerNorm(self.hidden_size)
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.embedding(input_ids)
        aux_total = torch.zeros(1, device=x.device, dtype=x.dtype)
        for blk in self.blocks:
            x, l_aux = blk(x)
            aux_total = aux_total + l_aux
        logits = self.lm_head(self.final_norm(x))
        return logits, aux_total.squeeze()
    

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tiny MoE baseline training (simple version)")
    p.add_argument("--config", type=str, default="configs/base.json")
    p.add_argument("--out_dir", type=str, default="results")
    p.add_argument("--run_name", type=str, default="")
    return p.parse_args()


def autocast_context(precision: str, device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def pick_device(config: Dict) -> torch.device:
    if config.get("device"):
        return torch.device(config["device"])
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    config = load_json(args.config)

    if not args.run_name:
        args.run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    out_dir = Path(args.out_dir)
    ensure_dir(str(out_dir))

    device = pick_device(config)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    set_seed(int(config.get("seed", 42)))

    precision = str(config.get("precision", "fp32")).lower()
    if device.type != "cuda":
        precision = "fp32"

    # save used config for reproducibility
    config_used = dict(config)
    config_used["precision_effective"] = precision
    save_json(str(out_dir / f"{args.run_name}_config_used.json"), config_used)

    metadata = {
        "run_name": args.run_name,
        "command": " ".join(sys.argv),
        "git_commit": get_git_commit_hash(),
        "env": get_env_info(),
        "device": str(device),
        "config_path": args.config,
    }
    print(json.dumps(metadata, indent=2))

    model = TinyMoELM(config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )

    train_steps = int(config["train_steps"])
    log_every = int(config["log_every"])
    moe_aux_loss_coef = float(config.get("moe_aux_loss_coef", 0.01))
    grad_clip = float(config.get("grad_clip", 1.0))

    dataloader = build_dataloader(config, rank=0, world_size=1)
    data_iter = iter(dataloader)

    model.train()
    for step in range(1, train_steps + 1):
        batch = next(data_iter)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast_context(precision, device):
            logits, l_aux = model(input_ids)
            ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            loss = ce + moe_aux_loss_coef * l_aux

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if step % log_every == 0:
            print(f"step={step} loss={loss.item():.4f} ce={ce.item():.4f} aux={float(l_aux):.4f}")

    done_path = out_dir / f"{args.run_name}_done.json"
    save_json(str(done_path), {"run_name": args.run_name, "steps": train_steps})
    print(f"Saved: {done_path}")


if __name__ == "__main__":
    main()
