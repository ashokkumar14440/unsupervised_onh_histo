from pathlib import Path, PurePath
import pickle
from typing import Dict, List, Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import inc.python_image_utilities.image_util as iutil

matplotlib.use("Agg")


PathLike = Union[str, Path, PurePath]


class ImageInfo:
    RGB_AND_SOBEL = "rgb_and_sobel"
    RGB = "rgb"
    SOBEL = "sobel"
    GRAY = "gray"
    SOBEL_H = "sobel_h"
    SOBEL_V = "sobel_v"

    CHANNEL_COUNT = "channel_count"
    SLICES = "slices"

    CHANNEL_INFO = {
        RGB_AND_SOBEL: {
            CHANNEL_COUNT: 5,
            SLICES: {RGB: slice(0, 3), SOBEL_H: slice(3, 4), SOBEL_V: slice(4, 5)},
        },
        RGB: {CHANNEL_COUNT: 3, SLICES: {RGB: slice(0, 3)}},
        SOBEL: {CHANNEL_COUNT: 2, SLICES: {SOBEL_H: slice(0, 1), SOBEL_V: slice(1, 2)}},
        GRAY: {CHANNEL_COUNT: 1, SLICES: {GRAY: slice(0, 1)}},
    }

    def __init__(self, is_rgb: bool, use_rgb: bool, do_sobelize: bool, input_size: int):
        if use_rgb:
            assert is_rgb
        if use_rgb and do_sobelize:
            t = self.RGB_AND_SOBEL
        elif use_rgb:
            t = self.RGB
        elif do_sobelize:
            t = self.SOBEL
        else:
            assert not is_rgb
            t = self.GRAY
        self._info = self.CHANNEL_INFO[t]
        self._sobel = do_sobelize
        self._rgb = is_rgb
        self._perceived_shape = (input_size, input_size)

    @property
    def channel_count(self):
        return self._info[self.CHANNEL_COUNT]

    @property
    def slices(self):
        return self._info[self.SLICES]

    @property
    def is_rgb(self):
        return self._rgb

    @property
    def sobel(self):
        return self._sobel

    @property
    def perceived_shape(self):
        return self._perceived_shape

    def check_input_image(self, image: np.array):
        # TODO figure out how to automate this
        if self.is_rgb:
            ok = image.ndim == 3
            c = image.shape[-1]
            ok = ok and c == 3
        else:
            ok = image.ndim == 2
        ok = ok and image.dtype == np.uint8
        return ok

    def check_output_image(self, image: np.array):
        ok = image.ndim == 3
        h, w, c = image.shape
        ok = ok and c == self.channel_count
        ok = ok and h == self.perceived_shape[0]
        ok = ok and w == self.perceived_shape[1]
        ok = ok and image.dtype == np.float32
        return ok

    def check_input_label(self, label: np.array):
        ok = label.ndim == 2
        ok = ok and label.dtype == np.int32
        return ok

    def check_output_label(self, label: np.array):
        ok = label.ndim == 3
        h, w, c = label.shape
        ok = ok and c == 1
        ok = ok and h == self.perceived_shape[0]
        ok = ok and w == self.perceived_shape[1]
        ok = ok and label.dtype == np.int32
        return ok


class OutputFiles:
    RENDER = "render"
    STATE = "state"
    SUBFOLDERS = [RENDER, STATE]

    def __init__(
        self, root_path: PathLike, image_info: ImageInfo, extension: str = ".png"
    ):
        root = PurePath(root_path)
        if not (Path(root).is_dir() and Path(root).exists()):
            Path(root).mkdir(parents=True, exist_ok=True)

        sub_root = {}
        for sub in self.SUBFOLDERS:
            path = root / sub
            if not (Path(path).is_dir() and Path(path).exists()):
                Path(path).mkdir(parents=True, exist_ok=True)
            sub_root[sub] = PurePath(path)

        self.root = root
        self._sub = sub_root
        self._ext = extension
        self._image_info = image_info

    def get_sub_root(self, key: str):
        return self._sub[key]

    def save_mask_tensor(self, name: str, image: torch.Tensor):
        out = self._to_numpy(image)
        self.save_label(name, out)

    def save_confidence_tensor(self, name: str, image: torch.Tensor):
        """
        confidence dimension first, then space
        """
        out = self._to_numpy(image)
        assert out.shape[0] <= 255
        out = out.argmax(axis=0).astype(np.uint8)
        self.save_label(name, out)

    def save_image(self, name: str, image: np.ndarray):
        image = image.copy()
        assert image.ndim == 3
        assert image.shape[-1] == self._image_info.channel_count
        for suffix, slc in self._image_info.slices.items():
            path = self._compose_render_path(name, suffix)
            iutil.save(path, image[:, :, slc])

    def save_label(self, name: str, label: np.ndarray):
        label = label.copy()
        if label.ndim == 3:
            assert label.shape[-1] == 1
            label = label.squeeze()
        assert label.ndim == 2
        path = self._compose_render_path(name, "label")
        label = iutil.rescale(label)
        label[label < 0] = 255
        iutil.save(path, label)

    def save_statistics_plots(
        self, name: str, data: pd.DataFrame, titles: Dict[str, str] = {}
    ):
        fig, axes = plt.subplots(len(data.columns), sharex=False, figsize=(20, 20))
        for column, ax in zip(data.columns, axes):
            ax.plot(data.loc[:, column])
            if column in titles:
                title = titles[column]
            else:
                title = column
            ax.set_title(title)
        fig.canvas.draw_idle()
        path = self._compose_render_path(name)
        fig.savefig(path)
        plt.close(fig)

    def _compose_render_path(
        self, name: str, suffix: Optional[str] = None, ext: Optional[str] = None
    ):
        if ext is None:
            ext = self._ext
        if not (suffix is None or suffix == ""):
            name = self._join_suffix(name, suffix)
        return self.get_sub_root(self.RENDER) / (name + ext)

    def _join_suffix(self, name: str, suffix: str):
        return "_".join([name, suffix])

    def _to_numpy(self, t: torch.Tensor) -> np.ndarray:
        return t.cpu().detach().numpy()


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

    def __init__(self, limit: int, output_files: OutputFiles):
        assert 0 < limit
        self._limit = limit
        self._output_files = output_files
        self._data = None

    @property
    def epochs(self):
        return range(self.count, self._limit)

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

    def add(self, values: dict, batch_statistics: List[BatchStatistics]):
        """
        keys in values and each batch_statistics must be globally unique
        """
        for stats in batch_statistics:
            means = stats.get_means()
            for k, v in means.items():
                assert k not in values
                values[k] = v
        if self._data is None:
            self._data = pd.DataFrame(columns=list(values.keys()))
        assert set(values.keys()) == set(self._data.columns)
        self._data.loc[self.count] = values

    def is_smaller(self, key: str, value: float) -> bool:
        if self._data is None:
            out = True
        else:
            out = value < self._data.loc[:, key].min()
        return out

    def draw(self, titles: Dict[str, str]):
        self._output_files.save_statistics_plots(
            "epoch_statistics", self._data.loc[:, list(titles.keys())], titles
        )

    def save_data(self):
        self._data.to_csv(
            self._output_files.root / "epochs.csv", index=False, sep=self._DELIMITER
        )

    def save(self, file_path: PathLike):
        with open(file_path, mode="wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file_path: PathLike):
        with open(file_path, mode="rb") as f:
            data = pickle.load(f)
        return data
