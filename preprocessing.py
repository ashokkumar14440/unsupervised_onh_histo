from typing import Optional

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as tvt

from src.utils.segmentation.transforms import (
    get_center_start_subscript,
    get_random_start_subscript,
    reshape_by_pad_crop,
    random_affine,
    custom_greyscale_numpy,
)
from src.utils.cluster.transforms import sobel_process


class Transformation:
    def __init__(
        self,
        rotation_range,
        shear_range,
        scale_range,
        flip_horizontal_probability,
        use_random_affine: bool,
    ):
        self._use_random_affine = random_affine
        self._rotation_range = self._prepare_range(rotation_range)
        self._shear_range = self._prepare_range(shear_range)
        self._scale_range = self._prepare_range(scale_range)
        self._flip_horizontal_probability = flip_horizontal_probability
        assert self._rotation_range.size == 2
        assert self._shear_range.size == 2
        assert self._scale_range.size == 2

    def apply(self, image):
        if self._use_random_affine:
            image, fwd, inv = self._transform_affine_random(image)
        else:
            image, fwd, inv = self._transform_identity(image)
        return image, fwd, inv

    def _transform_identity(self, image):
        forward_transform = torch.zeros([2, 3]).to(torch.float32).cuda()
        forward_transform[0, 0] = 1
        forward_transform[1, 1] = 1
        inverse_transform = torch.zeros([2, 3]).to(torch.float32).cuda()
        inverse_transform[0, 0] = 1
        inverse_transform[1, 1] = 1
        return image, forward_transform, inverse_transform

    def _transform_affine_random(self, image):
        """
        Transforms the input image and returns it, together with the forward and
        inverse transform arrays.
        """
        affine_kwargs = {
            "min_rot": self._rotation_range[0],
            "max_rot": self._rotation_range[1],
            "min_shear": self._shear_range[0],
            "max_shear": self._shear_range[1],
            "min_scale": self._scale_range[0],
            "max_scale": self._scale_range[1],
        }
        return random_affine(image, **affine_kwargs)

    def _flip_horizontal_random(self, img, affine_t_to_norm):
        if np.random.rand() > self._flip_horizontal_probability:
            img = torch.flip(img, dims=[2])
            affine_t_to_norm[0, :] *= -1.0
        return img, affine_t_to_norm

    @staticmethod
    def _prepare_range(values: list):
        v = np.array(values)
        v.sort(axis=-1)
        return v


class Preprocessing:
    def __init__(
        self,
        transformation: Transformation,
        input_size,
        include_rgb,
        prescale_all: bool,
        prescale_factor,
        jitter_brightness,
        jitter_contrast,
        jitter_saturation,
        jitter_hue,
        sobelize: bool,
    ):
        self._transformation = transformation
        self._perceived_shape = [input_size, input_size]
        self._include_rgb = include_rgb
        self._do_prescale = prescale_all
        self._prescale_factor = prescale_factor
        self._do_sobelize = sobelize
        self._jitter_brightness = jitter_brightness
        self._jitter_contrast = jitter_contrast
        self._jitter_saturation = jitter_saturation
        self._jitter_hue = jitter_hue

    def scale_data(self, image: np.array):
        return self._scale(image, dtype=np.float32, interp_mode=cv2.INTER_LINEAR)

    def scale_labels(self, image: np.array):
        return self._scale(image, dtype=np.int32, interp_mode=cv2.INTER_NEAREST)

    def force_dims(self, image):
        if image.ndim == 2:
            image = image[..., np.newaxis]
        return image

    def pad_crop(self, image: np.array, start_subscript) -> np.array:
        required_shape = self._get_required_shape(image)
        image = reshape_by_pad_crop(image, required_shape, start_subscript)
        return image

    def get_random_start_subscript(self, image: np.array):
        required_shape = self._get_required_shape(image)
        return get_random_start_subscript(image.shape, required_shape)

    def get_center_start_subscript(self, image: np.array):
        required_shape = self._get_required_shape(image)
        return get_center_start_subscript(image.shape, required_shape)

    def color_jitter(self, image: np.array):
        was_gray = False
        if image.shape[-1] == 1:
            was_gray = True
            image = image.squeeze()
        image = Image.fromarray(image.astype(np.uint8))
        image = tvt.ColorJitter(
            brightness=self._jitter_brightness,
            contrast=self._jitter_contrast,
            saturation=self._jitter_saturation,
            hue=self._jitter_hue,
        )
        image = np.array(image)
        if was_gray:
            image = image[..., np.newaxis]
        return image

    def grayscale(self, image: np.array):
        if self._use_rgb:
            assert image[-1] == 3
        if self._do_sobelize:
            # TODO this is nonsense, fix this function to be less opaque
            image = custom_greyscale_numpy(image, include_rgb=self._use_rgb)

    def scale_values(self, image: np.array):
        return image.astype(np.float32) / 255.0

    def torchify(self, image: np.array):
        return torch.from_numpy(image).permute(2, 0, 1).cuda()

    def transform(self, image: np.array):
        image, _, inverse = self._transformation.apply(image)
        return image, inverse

    def sobelize(self, image: np.array):
        if self._do_sobelize:
            image = sobel_process(img)

    def squeeze(self, image: np.array):
        return image.squeeze()

    def _scale(self, image: np.array, dtype, interp_mode):
        image = image.astype(dtype)
        image = cv2.resize(
            image,
            dsize=None,
            fx=self._prescale_factor,
            fy=self._prescale_factor,
            interpolation=interp_mode,
        )
        return image

    def _get_required_shape(self, image: np.array):
        required_shape = [*self._perceived_shape[0:2], *image.shape[2:]]
        return required_shape


class TrainImagePreprocessor:
    def __init__(self, preprocessing: Preprocessing):
        self._pre = preprocessing

    def apply(self, image: np.array, label: Optional[np.array] = None):
        t_image = image.copy()
        start_subscript = self._pre.get_random_start_subscript(image)

        image = self._pre.scale_data(image)
        image = self._pre.force_dims(image)
        image = self._pre.pad_crop(image, start_subscript)
        image = self._pre.grayscale(image)
        image = self._pre.scale_values(image)
        image = self._pre.torchify(image)
        image = self._pre.sobelize(image)

        t_image = self._pre.scale_data(t_image)
        t_image = self._pre.force_dims(t_image)
        t_image = self._pre.pad_crop(t_image, start_subscript)
        t_image = self._pre.color_jitter(t_image)
        t_image = self._pre.grayscale(t_image)
        t_image = self._pre.scale_values(t_image)
        t_image = self._pre.torchify(t_image)
        t_image, affine_inverse = self._pre.transform(t_image)
        t_image = self._pre.sobelize(t_image)

        if label is not None:
            label = self._pre.scale_labels(label)
            label = self._pre.force_dims(label)
            label = self._pre.pad_crop(label, start_subscript)
            label = self._pre.squeeze(label)

        return {
            "image": image,
            "label": label,
            "transformed_image": t_image,
            "affine_inverse": affine_inverse,
        }


class TestImagePreprocessor:
    def __init__(self, preprocessing: Preprocessing):
        self._pre = preprocessing

    def apply(self, image: np.array, label: Optional[np.array] = None):
        start_subscript = self._pre.get_center_start_subscript(image)

        image = self._pre.scale_data(image)
        image = self._pre.force_dims(image)
        image = self._pre.pad_crop(image, start_subscript)
        image = self._pre.grayscale(image)
        image = self._pre.scale_values(image)
        image = self._pre.torchify(image)
        image = self._pre.sobelize(image)

        if label is not None:
            label = self._pre.scale_labels(label)
            label = self._pre.force_dims(label)
            label = self._pre.pad_crop(label, start_subscript)
            label = self._pre.squeeze(label)

        return image


class EvalImagePreprocessor:
    def __init__(self, preprocessing: Preprocessing):
        self._pre = preprocessing

    def apply(self, image: np.array):
        image = self._pre.scale_data(image)
        image = self._pre.force_dims(image)
        image = self._pre.grayscale(image)
        image = self._pre.scale_values(image)
        image = self._pre.torchify(image)
        image = self._pre.sobelize(image)
        return image
