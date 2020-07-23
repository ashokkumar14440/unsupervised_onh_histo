from pathlib import Path, PurePath
from typing import Callable, Optional, Union

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as tvt

from transforms import (
    get_center_start_subscript,
    get_random_start_subscript,
    reshape_by_pad_crop,
    random_affine,
    sobel_process,
)
import utils

PathLike = Union[str, Path, PurePath]


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


class LabelMapper:
    def __init__(self, mapping_function: Optional[Callable] = None):
        self._mapping_function = mapping_function

    def apply(self, label: np.array):
        if self._mapping_function is not None:
            return self._mapping_function(label)
        else:
            return label


class SimplePreprocessing:
    def __init__(
        self, image_info: utils.ImageInfo, prescale_all: bool, prescale_factor: float
    ):
        self._image_info = image_info
        self._do_prescale = prescale_all
        self._prescale_factor = prescale_factor

    def scale_data(self, image: np.array):
        if self._do_prescale:
            out = self._scale(image, dtype=np.float32, interp_mode=cv2.INTER_LINEAR)
        else:
            out = image
        return out

    def scale_labels(self, image: np.array):
        if self._do_prescale:
            out = self._scale(image, dtype=np.int32, interp_mode=cv2.INTER_NEAREST)
        else:
            out = image
        return out

    def force_dims(self, image):
        if image.ndim == 2:
            image = image[..., np.newaxis]
        return image

    def grayscale(self, image: np.array):
        if self._image_info.sobel:
            image = self._to_grayscale(image)
        return image

    def scale_values(self, image: np.array):
        return image.astype(np.float32) / 255.0

    def torchify(self, image: np.array):
        return torch.from_numpy(image).permute(2, 0, 1).cuda()

    def sobelize(self, image: torch.Tensor):
        if self._image_info.sobel:
            image = sobel_process(image)
        return image

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

    def _to_grayscale(self, image: np.array):
        assert image.ndim == 3
        if self._image_info.is_rgb:
            h, w, c = image.shape
            assert c == 3
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).reshape(h, w, 1)
            image = np.concatenate([image, gray_image], axis=2)
        return image


class TransformPreprocessing(SimplePreprocessing):
    def __init__(
        self,
        transformation: Transformation,
        image_info: utils.ImageInfo,
        prescale_all: bool,
        prescale_factor,
        jitter_brightness,
        jitter_contrast,
        jitter_saturation,
        jitter_hue,
        label_mapper: Optional[LabelMapper] = None,
    ):
        if label_mapper is None:
            label_mapper = LabelMapper()
        self._jitter_fn = tvt.ColorJitter(
            brightness=jitter_brightness,
            contrast=jitter_contrast,
            saturation=jitter_saturation,
            hue=jitter_hue,
        )
        self._transformation = transformation
        self._label_mapper = label_mapper
        self._image_info = image_info
        self._do_prescale = prescale_all
        self._prescale_factor = prescale_factor

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
        image = self._jitter_fn(image)
        image = np.array(image)
        if was_gray:
            image = image[..., np.newaxis]
        return image

    def transform(self, image: np.array):
        image, _, inverse = self._transformation.apply(image)
        return image, inverse

    def map_labels(self, label: np.array):
        return self._label_mapper.apply(label)

    def _get_required_shape(self, image: np.array):
        required_shape = [*self._image_info.perceived_shape, *image.shape[2:]]
        return required_shape


class ImagePreprocessor:
    IMAGE = "image"
    LABEL = "label"

    def __init__(
        self,
        image_info: utils.ImageInfo,
        output_files: Optional[utils.OutputFiles] = None,
        do_render: bool = False,
        render_limit: int = 1,
    ):
        if do_render:
            assert output_files is not None

        self._image_info = image_info
        self._output = output_files
        self._render = do_render
        self._limit = render_limit
        self._count = 0

    def apply(self, **kwargs):
        out = self._apply_impl(**kwargs)
        self._increment()
        return out

    def _apply_impl(self, **kwargs):
        return {}

    def _save_image(self, name: str, image: np.array):
        if self._render and self._do_continue:
            self._output.save_image(name, image)

    def _save_label(self, name: str, label: np.array):
        if self._render and self._do_continue:
            self._output.save_label(name, label)

    def _increment(self):
        self._count += 1

    def _do_continue(self):
        return self._count < self._limit


class TrainImagePreprocessor(ImagePreprocessor):
    def __init__(self, preprocessing: TransformPreprocessing, **kwargs):
        super(TrainImagePreprocessor, self).__init__(**kwargs)
        self._pre = preprocessing

    def _apply_impl(
        self,
        file_path: PathLike,
        image: np.array,
        label: Optional[np.array] = None,
        **kwargs
    ):
        assert self._image_info.check_input_image(image)
        image = self._pre.scale_data(image)
        image = self._pre.force_dims(image)

        start_subscript = self._pre.get_random_start_subscript(image)
        image = self._pre.pad_crop(image, start_subscript)
        t_image = image.copy()

        image = self._pre.grayscale(image)
        image = self._pre.scale_values(image)
        image = self._pre.torchify(image)
        image = self._pre.sobelize(image)
        np_image = image.cpu().detach().numpy().transpose(1, 2, 0)
        assert self._image_info.check_output_image(np_image)
        name = PurePath(file_path).stem + "_train"
        self._save_image(name, np_image)

        t_image = self._pre.color_jitter(t_image)
        t_image = self._pre.grayscale(t_image)
        t_image = self._pre.scale_values(t_image)
        t_image = self._pre.torchify(t_image)
        t_image, affine_inverse = self._pre.transform(t_image)
        t_image = self._pre.sobelize(t_image)
        np_t_image = t_image.cpu().detach().numpy().transpose(1, 2, 0)
        assert self._image_info.check_output_image(np_t_image)
        t_name = name + "_t"
        self._save_image(t_name, np_t_image)

        out = {
            "image": image,
            "transformed_image": t_image,
            "affine_inverse": affine_inverse,
            "file_path": file_path,
            **kwargs,
        }

        if label is not None:
            assert self._image_info.check_input_label(label)
            label = self._pre.scale_labels(label)
            label = self._pre.force_dims(label)
            label = self._pre.pad_crop(label, start_subscript)
            label = self._pre.map_labels(label)
            assert self._image_info.check_output_label(label)
            self._save_label(name, label)
            label = self._pre.torchify(label)
            out["label"] = label

        return out


class TestImagePreprocessor(ImagePreprocessor):
    def __init__(self, preprocessing: TransformPreprocessing, **kwargs):
        super(TestImagePreprocessor, self).__init__(**kwargs)
        self._pre = preprocessing

    def _apply_impl(
        self,
        file_path: PathLike,
        image: np.array,
        label: Optional[np.array] = None,
        **kwargs
    ):
        assert self._image_info.check_input_image(image)
        image = self._pre.scale_data(image)
        image = self._pre.force_dims(image)

        start_subscript = self._pre.get_center_start_subscript(image)
        image = self._pre.pad_crop(image, start_subscript)
        image = self._pre.grayscale(image)
        image = self._pre.scale_values(image)
        image = self._pre.torchify(image)
        image = self._pre.sobelize(image)
        np_image = image.cpu().numpy().transpose(1, 2, 0)
        assert self._image_info.check_output_image(np_image)
        name = PurePath(file_path).stem + "_test"
        self._save_image(name, np_image)

        if label is not None:
            assert self._image_info.check_input_label(label)
            label = self._pre.scale_labels(label)
            label = self._pre.force_dims(label)
            label = self._pre.pad_crop(label, start_subscript)
            label = self._pre.map_labels(label)
            assert self._image_info.check_output_label(label)
            self._save_label(name, label)
            label = self._pre.torchify(label)

        return {"image": image, "label": label, "file_path": file_path, **kwargs}


class EvalImagePreprocessor(ImagePreprocessor):
    def __init__(self, preprocessing: SimplePreprocessing, **kwargs):
        super(EvalImagePreprocessor, self).__init__(**kwargs)
        self._pre = preprocessing

    def _apply_impl(self, image: np.ndarray, **kwargs):
        assert self._image_info.check_input_image(image)
        image = self._pre.scale_data(image)
        image = self._pre.force_dims(image)
        image = self._pre.grayscale(image)
        image = self._pre.scale_values(image)
        image = self._pre.torchify(image)
        image = self._pre.sobelize(image)
        np_image = image.cpu().numpy().transpose(1, 2, 0)
        assert self._image_info.check_output_eval_image(np_image)
        return {"image": image, **kwargs}
