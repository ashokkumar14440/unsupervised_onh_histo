import pickle
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import pandas as pd

from inc.python_file_utilities.file_utils import *
from src.utils.cluster.transforms import sobel_process


IMAGE_1 = 0
IMAGE_2 = 1
AFFINE = 2
MASK = 3


def transfer_images(loader_tuple, config):
    images = create_image_list(config)
    current_batch_size = get_dataloader_batch_size(loader_tuple)
    for index in range(config.dataset.num_dataloaders):
        img1, img2, affine2_to_1, mask_img1 = loader_tuple[index]
        assert img1.shape[0] == current_batch_size

        actual_batch_start = index * current_batch_size
        actual_batch_end = actual_batch_start + current_batch_size

        images[IMAGE_1][actual_batch_start:actual_batch_end, :, :, :] = img1
        images[IMAGE_2][actual_batch_start:actual_batch_end, :, :, :] = img2
        images[AFFINE][actual_batch_start:actual_batch_end, :, :] = affine2_to_1
        images[MASK][actual_batch_start:actual_batch_end, :, :] = mask_img1

        # if not (current_batch_size == config.dataloader_batch_sz) and (
        #     e_i == next_epoch
        # ):
        #     print("last batch sz %d" % curr_batch_sz)

        total_size = current_batch_size * config.dataset.num_dataloaders  # times 2
        images[IMAGE_1] = images[IMAGE_1][:total_size, :, :, :]
        images[IMAGE_2] = images[IMAGE_2][:total_size, :, :, :]
        images[AFFINE] = images[AFFINE][:total_size, :, :]
        images[MASK] = images[MASK][:total_size, :, :]
    return images


def sobelize(images, config):
    images[IMAGE_1] = sobel_process(
        images[IMAGE_1], config.include_rgb, using_IR=config.using_IR
    )
    images[IMAGE_2] = sobel_process(
        images[IMAGE_2], config.include_rgb, using_IR=config.using_IR
    )
    return images


def process(images, net, head):
    return [net(images[IMAGE_1], head=head), net(images[IMAGE_2], head=head)]


def compute_losses(config, loss_fn, lamb, images, outs):
    # averaging over heads
    avg_loss_batch = None
    avg_loss_no_lamb_batch = None

    for i in range(config.num_sub_heads):
        loss, loss_no_lamb = loss_fn(
            outs[IMAGE_1][i],
            outs[IMAGE_2][i],
            all_affine2_to_1=images[AFFINE],
            all_mask_img1=images[MASK],
            lamb=lamb,
            half_T_side_dense=config.half_T_side_dense,
            half_T_side_sparse_min=config.half_T_side_sparse_min,
            half_T_side_sparse_max=config.half_T_side_sparse_max,
        )

        if avg_loss_batch is None:
            avg_loss_batch = loss
            avg_loss_no_lamb_batch = loss_no_lamb
        else:
            avg_loss_batch += loss
            avg_loss_no_lamb_batch += loss_no_lamb

    avg_loss_batch /= config.num_sub_heads
    avg_loss_no_lamb_batch /= config.num_sub_heads
    return [avg_loss_batch, avg_loss_no_lamb_batch]


def create_image_list(config):
    return [
        create_empty(config),
        create_empty(config),
        create_empty_affine(config),
        create_empty_mask(config),
    ]


def create_empty(config):
    empty = torch.zeros(
        config.dataset.batch_size,
        get_channel_count(config),
        config.input_size,
        config.input_size,
    )
    return empty.to(torch.float32).cuda()


def create_empty_affine(config):
    empty = torch.zeros(config.dataset.batch_size, 2, 3)
    return empty.to(torch.float32).cuda()


def create_empty_mask(config):
    empty = torch.zeros(config.dataset.batch_size, config.input_size, config.input_size)
    return empty.to(torch.float32).cuda()


def get_channel_count(config):
    if config.preprocessor.sobelize:
        channel_count = config.in_channels - 1
    else:
        channel_count = config.in_channels
    return channel_count


def get_dataloader_batch_size(loader_tuple):
    return loader_tuple[0][0].shape[0]


class Canvas:
    _COUNT = 6
    _TITLES = [
        "Accuracy, Best (Top: {top:.3f})",
        "Accuracy, Average (Top: {top:.3f})",
        "Loss, Head A",
        "Loss, Head A, No Lamb",
        "Loss, Head B",
        "Loss, Head B, No Lamb",
    ]
    _FIELDS = [
        "acc",
        "avg_subhead_acc",
        "loss_A",
        "loss_no_lamb_A",
        "loss_B",
        "loss_no_lamb_B",
    ]

    def __init__(self):
        assert self._COUNT == len(self._TITLES) == len(self._FIELDS)
        self._fig, self._axarr = plt.subplots(
            self._COUNT, sharex=False, figsize=(20, 20)
        )

    def draw(self, epoch_statistics: "EpochStatistics"):
        for index in range(self._COUNT):
            self._draw_plot(epoch_statistics, index)
        self._fig.canvas.draw_idle()

    def save(self, file_path):
        self._fig.savefig(file_path)

    def _draw_plot(self, epoch_statistics, index):
        self._axarr[index].clear()
        self._axarr[index].plot(epoch_statistics[self._FIELDS[index]])
        self._axarr[index].set_title(self._TITLES[index])


class BatchStatistics:
    def __init__(self):
        self._data = None

    @property
    def count(self):
        if self._data is None:
            return 0
        else:
            return len(self._data)

    def add(self, values: dict):
        names = list(values.keys())
        if self._data is None:
            self._data = pd.DataFrame(columns=names)
        assert set(names) == set(self._data.columns)
        self._data.loc[self.count] = values

    def get_means(self):
        return dict(self._data.mean())


class EpochStatistics:
    _DELIMITER = "|"

    def __init__(self):
        self._data = None

    @property
    def count(self):
        if self._data is None:
            return 0
        else:
            return len(self._data)

    def __contains__(self, key):
        if self._data is None:
            return False
        else:
            return key in self._data.columns

    def __getitem__(self, key):
        return list(self._data.loc[:, key])

    def add(self, values: dict, batch_statistics: Dict[str, BatchStatistics]):
        all_means = {}
        for head, stats in batch_statistics.items():
            means = stats.get_means()
            names = {old: "_".join([old, head]) for old in means.keys()}
            all_means.update({new: means[old] for old, new in names.items()})
        names = [*values.keys(), *all_means.keys()]
        if self._data is None:
            self._data = pd.DataFrame(columns=names)
        assert set(names) == set(self._data.columns)
        self._data.loc[self.count] = {**values, **all_means}

    def save(self, path_or_file_obj):
        assert self._data is not None
        self._data.to_csv(path_or_file_obj, index=False, sep=self._DELIMITER)

    def load(self, path_or_file_obj):
        self._data = pd.read_csv(path_or_file_obj, sep=self._DELIMITER)

    @classmethod
    def from_file(cls, path_or_file_obj):
        instance = cls()
        instance.load(path_or_file_obj)
        return instance


def save_text(text, f):
    f.write(str(text))


def save_stats(stats: EpochStatistics, f):
    stats.save(f)


class StateFiles:
    _SAVE_STRATEGIES = {
        "pytorch": {
            "ext": ".pytorch",
            "infix": "",
            "binary_mode": "b",
            "load_fn": torch.load,
            "save_fn": torch.save,
        },
        "config_binary": {
            "ext": ".pickle",
            "infix": "config",
            "binary_mode": "b",
            "load_fn": pickle.load,
            "save_fn": pickle.dump,
        },
        "config_readable": {
            "ext": ".txt",
            "infix": "config",
            "binary_mode": "",
            "load_fn": None,
            "save_fn": save_text,
        },
        "statistics": {
            "ext": ".csv",
            "infix": "",
            "binary_mode": "",
            "load_fn": EpochStatistics.from_file,
            "save_fn": save_stats,
        },
    }

    def __init__(self, config):
        path = Path(config.out_dir).resolve()
        if not (path.is_dir() and path.exists()):
            raise ValueError("Could not locate output folder.")
        self._base = Files(path)

    def exists(self, state_file, suffix):
        name = self._get_file_name(state_file, suffix)
        return Path(name).is_file() and Path(name).exists()

    def load(self, state_file, suffix):
        with self._prepare_context(state_file, suffix, "r") as f:
            data = self._SAVE_STRATEGIES[state_file]["load_fn"](f)
        return data

    def save(self, state_file, suffix, data):
        with self._prepare_context(state_file, suffix, "w") as f:
            self._SAVE_STRATEGIES[state_file]["save_fn"](data, f)

    def save_state(self, suffix, config, pytorch_data, stats):
        self.save("pytorch", suffix, pytorch_data)
        self.save("config_binary", suffix, config)
        self.save("config_readable", suffix, config)
        self.save("statistics", suffix, stats)

    def _prepare_context(self, state_file, suffix, mode):
        binary_mode = self._SAVE_STRATEGIES[state_file]["binary_mode"]
        return open(self._get_file_name(state_file, suffix), mode + binary_mode)

    def _get_file_name(self, state_file: str, suffix: str):
        s = self._SAVE_STRATEGIES[state_file]
        infix = s["infix"]
        files = self._base + infix + suffix
        ext = s["ext"]
        return files.generate_file_names(ext=ext)[0]
