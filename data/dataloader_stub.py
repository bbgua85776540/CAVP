import torch
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.distributed import DistributedSampler

from .build import build_dataset

def get_train_val_loaders(cfg):
    train_loader = construct_loader(cfg, 'train')
    val_loader = construct_loader(cfg, 'val')
    return train_loader, val_loader

def construct_loader(cfg, split):
    """
    Constructs the data loader for the given dataset.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        split (str): the split of the data loader. Options include `train`,
            `val`, and `test`.
    """
    assert split in ["train", "val", "test"]
    if split in ["train"]:
        dataset_name = 'vrdataset'
        batch_size = int(cfg.batch_size / max(1, 1))
        shuffle = True
        drop_last = True
        # shuffle = False
        # drop_last = False
    elif split in ["val"]:
        dataset_name = 'evaldataset'
        batch_size = int(cfg.batch_size / max(1, 1))
        shuffle = False
        drop_last = True
    elif split in ["test"]:
        dataset_name = 'evaldataset'
        batch_size = int(cfg.eval_batch_size / max(1, 1))
        shuffle = False
        drop_last = False

    # Construct the dataset
    dataset = build_dataset(dataset_name, cfg, split)
    # print('dataset=', dataset)

    # Create a sampler for multi-process training
    sampler = DistributedSampler(dataset) if 1 > 1 else None


    collate_func = None
    # print('dataset=', len(dataset))
    # print('batch_size=', batch_size)
    # Create a loader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(False if sampler else shuffle),
        sampler=sampler,
        num_workers=12,
        pin_memory=True,
        drop_last=drop_last,
        collate_fn=collate_func,
    )
    return loader


