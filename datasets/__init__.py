from torch.utils.data import DataLoader
from .rec_dataset import RecTextLineDataset
from .rec_collatefn import RecCollateFn


__all__ = ['build_rec_dataloader']


def build_rec_dataset(data_dir, label_file_list, character):
    return RecTextLineDataset(data_dir, label_file_list, character)


def build_rec_collate_fn():
    return RecCollateFn()


def build_rec_dataloader(data_dir, label_file_list, batchsize, num_workers, character,  is_train=False):
    dataset = build_rec_dataset(
        data_dir, label_file_list, character)

    collate_fn = build_rec_collate_fn()
    if is_train:
        loader = DataLoader(dataset=dataset, batch_size=batchsize,
                            collate_fn=collate_fn, shuffle=True, num_workers=num_workers)
    else:
        loader = DataLoader(dataset=dataset, batch_size=batchsize,
                            collate_fn=collate_fn, shuffle=False, num_workers=num_workers)
    return loader
