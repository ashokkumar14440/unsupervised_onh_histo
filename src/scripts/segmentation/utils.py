import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
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
    for index in range(config.num_dataloaders):
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

        total_size = current_batch_size * config.num_dataloaders  # times 2
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
        config.batch_sz, get_channel_count(config), config.input_sz, config.input_sz
    )
    return empty.to(torch.float32).cuda()


def create_empty_affine(config):
    empty = torch.zeros(config.batch_sz, 2, 3)
    return empty.to(torch.float32).cuda()


def create_empty_mask(config):
    empty = torch.zeros(config.batch_sz, config.input_sz, config.input_sz)
    return empty.to(torch.float32).cuda()


def get_channel_count(config):
    if not config.no_sobel:
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
        "epoch_acc",
        "epoch_avg_subhead_acc",
        "epoch_loss_head_A",
        "epoch_loss_no_lamb_head_A",
        "epoch_loss_head_B",
        "epoch_loss_no_lamb_head_B",
    ]

    def __init__(self):
        assert self._COUNT == len(self._TITLES) == len(self._FIELDS)
        self._fig, self._axarr = plt.subplots(
            self._COUNT, sharex=False, figsize=(20, 20)
        )

    def draw(self, config):
        for index in range(self._COUNT):
            self._draw_plot(config, index)
        self._fig.canvas.draw_idle()

    def save(self, file_path):
        self._fig.savefig(file_path)

    def _draw_plot(self, config, index):
        self._axarr[index].clear()
        self._axarr[index].plot(config.__dict__[self._FIELDS[index]])
        self._axarr[index].set_title(self._TITLES[index])


class Model:
    def __init__(self, state_files):
        pass

    @property
    def finished(self):
        pass

    def process_epoch(self):
        pass

    def _process_head(self):
        pass

    def _process_batch(self):
        pass

    def _print_status(self):
        pass


class BatchStatistics:
    # TODO replaces b_i, avg_loss, avg_loss_no_lamb, avg_loss_count
    def __init__(self, loss_names: Sequence[str]):
        self._data = pd.DataFrame(columns=loss_names)

    @property
    def count(self):
        return len(self._data)

    def add(self, loss_values: dict):
        assert set(loss_values.keys()) == set(self._data.columns)
        self._data.loc[self.count] = loss_values

    def get_means(self):
        return dict(self._data.mean())


class EpochStatistics:
    # TODO replaces epoch_acc, epoch_avg_subhead_acc, epoch_stats
    # TODO epoch_loss_head_A, epoch_loss_no_lamb_head_A
    # TODO epoch_loss_head_B, epoch_loss_no_lamb_head_B
    # TODO all above are config.*
    # TODO epoch_loss, epoch_loss_no_lamb
    def __init__(self):
        pass

    def add(self, batch_values):
        pass

    def print(self):
        pass


class StateFiles:
    _PICKLE = ".pickle"
    _TXT = ".txt"
    _PYTORCH = ".pytorch"
    _CONFIG = "config"

    def __init__(self, config):
        path = Path(config.out_dir).resolve()
        if not path.is_dir():
            raise ValueError("Could not locate output folder.")
        self._base = Files(path)

    def exists_config(self, suffix):
        files = self._get_config_files(suffix)
        name = self._get_file_name(files, self._PICKLE)
        return Path(name).is_file()

    def save_state(self, suffix, config, pytorch_data):
        self.save_pytorch(suffix, pytorch_data)
        self.save_config(suffix, config)

    def save_config(self, suffix, config):
        files = self._get_config_files(suffix)
        with open(self._get_file_name(files, self._PICKLE), "wb") as f:
            pickle.dump(config, f)
        with open(self._get_file_name(files, self._TXT), "w") as f:
            f.write(str(config))

    def load_config(self, suffix):
        files = self._get_config_files(suffix)
        with open(self._get_file_name(files, self._PICKLE), "rb") as f:
            config = pickle.load(f)
        return config

    def save_pytorch(self, suffix, pytorch_data):
        files = self._get_pytorch_files(suffix)
        torch.save(pytorch_data, self._get_file_name(files, self._PYTORCH))

    def load_pytorch(self, suffix):
        files = self._get_pytorch_files(suffix)
        return torch.load(self._get_file_name(files, self._PYTORCH))

    def _get_file_name(self, files, ext):
        return files.generate_file_names(ext=ext)[0]

    def _get_config_files(self, suffix):
        return self._base + self._CONFIG + suffix

    def _get_pytorch_files(self, suffix):
        return self._base + suffix
