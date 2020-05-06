import torch
from torch.utils.data import ConcatDataset

from src.scripts.segmentation import data
from src.scripts.segmentation.general_data import OnhDataset


def build_dataloaders(config, preprocessor):
    name = config.dataset.name
    if name in data.__dict__.keys():
        dataset_class = data.__dict__[config.dataset.name]
    else:
        dataset_class = OnhDataset

    preprocessor.purpose = "train"
    dataloaders = _create_dataloaders(
        config, dataset_class, preprocessor
    )  # type: ignore

    preprocessor.purpose = "test"
    mapping_assignment_dataloader = _create_mapping_loader(
        config,
        dataset_class,  # type: ignore
        config.dataset.partitions.map_assign,
        preprocessor,
    )

    preprocessor.purpose = "test"
    mapping_test_dataloader = _create_mapping_loader(
        config,
        dataset_class,  # type: ignore
        config.dataset.partitions.map_test,
        preprocessor,
    )

    return dataloaders, mapping_assignment_dataloader, mapping_test_dataloader


def _create_dataloaders(config, dataset_class, preprocessor):
    # unlike in clustering, each dataloader here returns pairs of images - we
    # need the matrix relation between them
    dataloaders = []
    do_shuffle = config.training.shuffle
    count = config.training.num_dataloaders
    for d_i in range(count):
        print("Creating dataloader {:d}/{:d}".format(d_i + 1, count))
        train_dataloader = torch.utils.data.DataLoader(
            _create_dataset(config, dataset_class, preprocessor),
            batch_size=config.dataloader_batch_size,
            shuffle=do_shuffle,
            num_workers=0,
            drop_last=False,
        )
        if d_i > 0:
            assert len(train_dataloader) == len(dataloaders[d_i - 1])
        dataloaders.append(train_dataloader)

    print(("Number of batches per epoch: {:d}".format(len(dataloaders[0]))))
    return dataloaders


def _create_mapping_loader(config, dataset_class, partitions, preprocessor):
    return torch.utils.data.DataLoader(
        _create_dataset(config, dataset_class, preprocessor),
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )


def _create_dataset(config, dataset_class, preprocessor):
    images = [
        dataset_class(
            config=config,
            split=partition,
            purpose=preprocessor.purpose,
            preprocessor=preprocessor,
        )
        for partition in config.dataset.partitions.train.values()
    ]
    return ConcatDataset(images)
