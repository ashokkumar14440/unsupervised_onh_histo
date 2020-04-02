import sys
from datetime import datetime

import torch
from torch.utils.data import ConcatDataset

from src.scripts.segmentation import data


def segmentation_create_dataloaders(config):
    # if config.mode == "IID+":
    #     if "Coco10k" in config.dataset:
    #         config.train_partitions = ["train"]
    #         config.mapping_assignment_partitions = ["train"]
    #         config.mapping_test_partitions = ["test"]
    #     elif "Coco164k" in config.dataset:
    #         config.train_partitions = ["train2017"]
    #         config.mapping_assignment_partitions = ["train2017"]
    #         config.mapping_test_partitions = ["val2017"]
    #     elif config.dataset == "Potsdam":
    #         config.train_partitions = ["unlabelled_train", "labelled_train"]
    #         config.mapping_assignment_partitions = ["labelled_train"]
    #         config.mapping_test_partitions = ["labelled_test"]
    #     else:
    #         raise NotImplementedError

    # elif config.mode == "IID":
    #     if "Coco10k" in config.dataset:
    #         config.train_partitions = ["all"]
    #         config.mapping_assignment_partitions = ["all"]
    #         config.mapping_test_partitions = ["all"]
    #     elif "Coco164k" in config.dataset:
    #         config.train_partitions = ["train2017", "val2017"]
    #         config.mapping_assignment_partitions = ["train2017", "val2017"]
    #         config.mapping_test_partitions = ["train2017", "val2017"]
    #     elif config.dataset == "Potsdam":
    #         config.train_partitions = [
    #             "unlabelled_train",
    #             "labelled_train",
    #             "labelled_test",
    #         ]
    #         config.mapping_assignment_partitions = ["labelled_train", "labelled_test"]
    #         config.mapping_test_partitions = ["labelled_train", "labelled_test"]
    #     else:
    #         raise NotImplementedError

    if "Coco" in config.dataset.name:
        dataloaders, mapping_assignment_dataloader, mapping_test_dataloader = make_Coco_dataloaders(
            config
        )
    else:
        raise NotImplementedError

    return dataloaders, mapping_assignment_dataloader, mapping_test_dataloader


def make_Coco_dataloaders(config):
    dataset_class = data.__dict__[config.dataset.name]
    dataloaders = _create_dataloaders(config, dataset_class, "train")  # type: ignore

    mapping_assignment_dataloader = _create_mapping_loader(
        config,
        dataset_class,  # type: ignore
        partitions=config.dataset.partitions.map_assign,
    )

    mapping_test_dataloader = _create_mapping_loader(
        config,
        dataset_class,  # type: ignore
        partitions=config.dataset.partitions.map_test,
    )

    return dataloaders, mapping_assignment_dataloader, mapping_test_dataloader


def _create_dataloaders(config, dataset_class, purpose):
    # unlike in clustering, each dataloader here returns pairs of images - we
    # need the matrix relation between them
    dataloaders = []
    do_shuffle = config.dataset.num_dataloaders == 1
    count = config.dataset.num_dataloaders
    for d_i in range(count):
        print("Creating dataloader {:d}/{:d}".format(d_i + 1, count))
        train_dataloader = torch.utils.data.DataLoader(
            _create_dataset(config, dataset_class, purpose),
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


def _create_dataset(config, dataset_class, purpose):
    train_images = [
        dataset_class(**{"config": config, "split": partition, "purpose": purpose})
        for partition in config.dataset.partitions.train
    ]
    return ConcatDataset(train_images)


def _create_mapping_loader(config, dataset_class, partitions):
    imgs_list = []
    for partition in partitions:
        imgs_curr = dataset_class(
            **{
                "config": config,
                "split": partition,
                "purpose": "test",
            }  # return testing tuples, image and label
        )
        imgs_list.append(imgs_curr)

    imgs = ConcatDataset(imgs_list)
    dataloader = torch.utils.data.DataLoader(
        imgs,
        batch_size=config.dataset.batch_size,
        # full batch
        shuffle=False,
        # no point since not trained on
        num_workers=0,
        drop_last=False,
    )
    return dataloader
