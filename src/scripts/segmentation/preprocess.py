import cv2
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data.dataset import ConcatDataset
import torchvision.transforms as tvt
from PIL import Image
from torch.utils import data
from ...utils.segmentation.transforms import (
    pad_and_or_crop,
    random_affine,
    custom_greyscale_numpy,
)


class Preprocessor:
    def __init__(self, config, purpose: str):
        assert purpose in ["train", "test"]
        self._purpose = purpose

        self._input_size = config.input_size
        self._include_rgb = config.include_rgb

        self._prescale_all = config.preprocessor.prescale_all
        self._prescale_factor = config.preprocessor.prescale_factor
        self._sobelize = config.preprocessor.sobelize
        self._jitter_tf = tvt.ColorJitter(
            brightness=config.preprocessor.jitter_brightness,
            contrast=config.preprocessor.jitter_contrast,
            saturation=config.preprocessor.jitter_saturation,
            hue=config.preprocessor.jitter_hue,
        )

        self._use_random_stretch = config.transformations.use_random_stretch
        self._scale_range = self._prepare_range(config.transformations.stretch_range)
        assert self._scale_range.size == 2
        self._flip_probability = config.transformations.flip_probability
        self._use_random_affine = config.transformations.use_random_affine
        self._rotation_range = self._prepare_range(
            config.transformations.rotation_range
        )
        assert self._rotation_range.size == 2
        self._shear_range = self._prepare_range(config.transformations.shear_range)
        assert self._shear_range.size == 2
        self._scale_range = self._prepare_range(config.transformations.scale_range)
        assert self._scale_range.size == 2

        assert self._prescale_factor < 1.0

    def process(self, image, label):
        if self._purpose == "train":
            img = self._preprocess_general_train(image, np.float32, cv2.INTER_LINEAR)
            lbl = self._preprocess_general_train(label, np.int32, cv2.INTER_NEAREST)
            img, lbl = self._pad_and_crop(img, lbl, ("random", "fixed"))
            t_img = np.copy(img)
            t_img, affine_inverse = self._preprocess_transformed(t_img)
            img = self._preprocess_train(img)
            return {
                "image": img,
                "label": lbl,
                "transformed_image": t_img,
                "affine_inverse": affine_inverse,
            }
        elif self._purpose == "train_single":
            img = self._preprocess_general_train(image, np.float32, cv2.INTER_LINEAR)
            lbl = self._preprocess_general_train(label, np.int32, cv2.INTER_NEAREST)
            img, lbl = self._pad_and_crop(img, lbl, ("random", "fixed"))
            img, _ = self._preprocess_transformed(img)
            return {"image": img, "label": label}
        elif self._purpose == "test":
            img = self._preprocess_general_train(image, np.float32, cv2.INTER_LINEAR)
            lbl = self._preprocess_general_train(label, np.int32, cv2.INTER_NEAREST)
            img, lbl = self._pad_and_crop(img, lbl, ("centre", "centre"))
            img, _ = self._preprocess_transformed(img)
            return {"image": img, "label": label}
        else:
            assert False

    def _preprocess_general_train(self, img, dtype, interpolation):
        img = self._preprocess_general_test(img, dtype, interpolation)
        img = self._stretch_random(img, interpolation)
        return img

    def _preprocess_general_test(self, img, dtype, interpolation):
        img = img.astype(dtype)
        img = self._prescale(img, interpolation)
        return img

    def _prescale(self, img, interpolation):
        if self._prescale_all:
            img = cv2.resize(
                img,
                dsize=None,
                fx=self._prescale_factor,
                fy=self._prescale_factor,
                interpolation=interpolation,
            )
        return img

    def _stretch_random(self, img, interpolation):
        if self._use_random_stretch:
            # bilinear interp requires float img
            stretch_factor = (
                np.random.rand() * (self._scale_range.diff())
            ) + self._scale_range[0]
            img = cv2.resize(
                img,
                dsize=None,
                fx=stretch_factor,
                fy=stretch_factor,
                interpolation=interpolation,
            )
        return img

    def _preprocess_train(self, img):
        img = self._handle_sobel(img)
        img = self._rescale_values(img)
        img = self._prepare_torch(img)
        return img

    def _preprocess_transformed(self, img):
        img = self._jitter(img)
        img = self._preprocess_train(img)
        img, affine_t_to_norm = self._transform_affine_random(img)
        img, affine_t_to_norm = self._flip_random(img, affine_t_to_norm)
        return img, affine_t_to_norm

    def _rescale_values(self, img):
        return img.astype(np.float32) / 255.0

    def _prepare_torch(self, img):
        return torch.from_numpy(img).permute(2, 0, 1).cuda()

    def _handle_sobel(self, img):
        if self._sobelize:
            img = custom_greyscale_numpy(img, include_rgb=self._include_rgb)
        return img

    def _pad_and_crop(self, img, label, modes: tuple):
        RAND_FIXED = ("random", "fixed")
        CENTER_CENTER = ("centre", "centre")
        assert modes in [RAND_FIXED, CENTER_CENTER]
        img, coords = pad_and_or_crop(img, self._input_size, mode=modes[0])
        if modes[1] != "fixed":
            coords = None
        label, _ = pad_and_or_crop(
            label, self._input_size, mode=modes[1], coords=coords
        )
        return img, label

    def _jitter(self, img):
        img = Image.fromarray(img.astype(np.uint8))
        img = self._jitter_tf(img)
        img = np.array(img)
        return img

    def _transform_affine_random(self, t_img):
        if self._use_random_affine:
            affine_kwargs = {
                "min_rot": self._rotation_range[0],
                "max_rot": self._rotation_range[1],
                "min_shear": self._shear_range[0],
                "max_shear": self._shear_range[1],
                "min_scale": self._scale_range[0],
                "max_scale": self._scale_range[1],
            }
            t_img, _, affine_t_to_norm = random_affine(t_img, **affine_kwargs)
        else:
            affine_t_to_norm = torch.zeros([2, 3]).to(torch.float32).cuda()
            affine_t_to_norm[0, 0] = 1
            affine_t_to_norm[1, 1] = 1
        return t_img, affine_t_to_norm

    def _flip_random(self, img, affine_t_to_norm):
        if np.random.rand() > self._flip_probability:
            img = torch.flip(img, dims=[2])
            affine_t_to_norm[0, :] *= -1.0
        return img, affine_t_to_norm

    @staticmethod
    def _prepare_range(values: list):
        v = np.array(values)
        v.sort(axis=-1)
        return v
