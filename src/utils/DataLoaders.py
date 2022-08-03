from torch.utils.data import DataLoader


def collate_fn(batch):
    return tuple(zip(*batch))


def get_loader(dataset, batch_size, shuffle, pin_memory=False, drop_last=False):
    params = {
        'dataset': dataset,
        'batch_size': batch_size,
        'shuffle': shuffle,
        'collate_fn': collate_fn,
        'drop_last': drop_last,
        'pin_memory': pin_memory
    }
    return DataLoader(**params)
