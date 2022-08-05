from torch.utils.data import DataLoader


def get_loader(dataset, batch_size, shuffle, collate_fn=None, pin_memory=False, drop_last=False):
    params = {
        'dataset': dataset,
        'batch_size': batch_size,
        'shuffle': shuffle,
        'collate_fn': collate_fn,
        'drop_last': drop_last,
        'pin_memory': pin_memory
    }
    return DataLoader(**params)
