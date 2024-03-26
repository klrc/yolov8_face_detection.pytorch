import os
import torch
from torch.utils.data import DataLoader


class RepeatSampler:
    """Sampler that repeats forever
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


def _check_num_workers(batch_size, num_workers):
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, num_workers])  # number of workers
    return nw


class InfiniteDataLoader(DataLoader):
    """Dataloader that reuses workers
    Uses same syntax as vanilla DataLoader
    """

    def __init__(
        self,
        dataset,
        training,
        collate_fn,
        batch_size=64,
        num_workers=8,
        pin_memory=False,
    ):
        super().__init__(
            dataset,
            batch_size=min(batch_size, len(dataset)),
            shuffle=training,
            sampler=None,
            num_workers=_check_num_workers(batch_size, num_workers),
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )
        object.__setattr__(self, "batch_sampler", RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self.iterator)
