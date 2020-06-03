from typing import Union
from pathlib import Path, PurePath
import numpy as np
from PIL import Image

import inc.python_image_utilities.image_util as iutil


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
    def __init__(
        self,
        root_path: PathLike,
        render_subfolder: str,
        image_info: ImageInfo,
        extension: str = ".png",
    ):
        root = PurePath(root_path)
        render_root = root / render_subfolder
        if not (Path(render_root).is_dir() and Path(render_root).exists()):
            Path(render_root).mkdir(parents=True, exist_ok=True)

        self._root = root
        self._render_root = render_root
        self._ext = extension
        self._image_info = image_info

    def save_image(self, name: str, image: np.array):
        image = image.copy()
        assert image.ndim == 3
        assert image.shape[-1] == self._image_info.channel_count
        for suffix, slc in self._image_info.slices.items():
            path = self._compose_render_path(name, suffix)
            iutil.save(path, image[:, :, slc])

    def save_label(self, name: str, label: np.array):
        label = label.copy()
        if label.ndim == 3:
            assert label.shape[-1] == 1
            label = label.squeeze()
        assert label.ndim == 2
        path = self._compose_render_path(name, "label")
        label = iutil.rescale(label)
        label[label < 0] = 255
        iutil.save(path, label)

    def _compose_render_path(self, name: str, suffix: str):
        name = "_".join([name, suffix])
        return self._render_root / (name + self._ext)
