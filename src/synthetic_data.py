from typing import Dict

import torch
from torch.utils.data import DataLoader, IterableDataset


class SyntheticTokenDataset(IterableDataset):
    def __init__(self, vocab_size: int, seq_len: int, batch_size: int, seed: int, rank: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.seed = seed
        self.rank = rank

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.rank)
        while True:
            input_ids = torch.randint(
                low=0,
                high=self.vocab_size,
                size=(self.batch_size, self.seq_len),
                generator=generator,
                dtype=torch.long,
            )
            yield {"input_ids": input_ids, "labels": input_ids.clone()}


def build_dataloader(config: Dict, rank: int, world_size: int) -> DataLoader:
    _ = world_size 
    dataset = SyntheticTokenDataset(
        vocab_size=int(config["vocab_size"]),
        seq_len=int(config["seq_len"]),
        batch_size=int(config["micro_batch_size"]),
        seed=int(config["seed"]),
        rank=rank,
    )
    return DataLoader(dataset, batch_size=None, num_workers=0, pin_memory=torch.cuda.is_available())

