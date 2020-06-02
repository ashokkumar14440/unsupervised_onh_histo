import cv2
import numpy as np
import torch
import torchvision.transforms as tvt
from PIL import Image
from src.utils.segmentation.transforms import (
    get_center_start_subscript,
    get_random_start_subscript,
    reshape_by_pad_crop,
    random_affine,
    custom_greyscale_numpy,
)
from src.utils.cluster.transforms import sobel_process


class Preparer:
    _IMAGE_TYPE_DATA = {
        "data": {"interp_mode": cv2.INTER_LINEAR, "dtype": np.float32},
        "label": {"interp_mode": cv2.INTER_NEAREST, "dtype": np.int32},
    }

    def __init__(self, config):
        self._prescale = config.preprocessor.prescale_all
        self._prescale_factor = config.preprocessor.prescale_factor
        assert self._prescale_factor < 1.0

    def apply_to_image(self, image):
        type_data = self._IMAGE_TYPE_DATA["data"]
        return self._apply(image, type_data)

    def apply_to_labels(self, image):
        type_data = self._IMAGE_TYPE_DATA["label"]
        return self._apply(image, type_data)

    def _apply(self, image, type_data):
        image = image.astype(type_data["dtype"])
        if self._prescale:
            image = self._scale(image, type_data)
        if image.ndim == 2:
            image = image[..., np.newaxis]
        return image

    def _scale(self, image, type_data):
        return cv2.resize(
            image,
            dsize=None,
            fx=self._prescale_factor,
            fy=self._prescale_factor,
            interpolation=type_data["interp_mode"],
        )


class Preprocessor:
    def __init__(self, config, preparer, transformation):
        self.do_render = False

        self._purpose = None
        self._preparer = preparer
        self._transformation = transformation

        self._include_rgb = config.dataset.parameters.use_rgb
        self._do_sobelize = config.preprocessor.sobelize
        input_size = config.dataset.parameters.input_size
        self._input_shape = [input_size, input_size]

        self._jitter_tf = tvt.ColorJitter(
            brightness=config.preprocessor.jitter_brightness,
            contrast=config.preprocessor.jitter_contrast,
            saturation=config.preprocessor.jitter_saturation,
            hue=config.preprocessor.jitter_hue,
        )

    @property
    def purpose(self):
        return self._purpose

    @purpose.setter
    def purpose(self, value):
        assert value in ["train", "test"]
        self._purpose = value

    def apply(self, image, label):
        assert self._purpose is not None
        img = self._preparer.apply_to_image(image)
        lbl = self._preparer.apply_to_labels(label)
        t_img = None
        affine_inverse = None
        if self._purpose == "train":
            img, lbl = self._pad_and_crop_random(img, lbl)
            t_img = np.copy(img)
            t_img, affine_inverse = self._preprocess_transformed(t_img)
        elif self._purpose == "train_single":
            img, lbl = self._pad_and_crop_random(img, lbl)
            img, _ = self._preprocess_transformed(img)
        elif self._purpose == "test":
            img, lbl = self._pad_and_crop_center(img, lbl)
        else:
            assert False
        assert img is not None
        assert lbl is not None

        img = self._preprocess(img)
        # lbl = self._prepare_torch(lbl)
        lbl = lbl.squeeze()
        if self._do_sobelize:
            img = self._sobelize(img)
            t_img = self._sobelize(t_img)
        return {
            "image": img,
            "label": lbl,
            "transformed_image": t_img,
            "affine_inverse": affine_inverse,
        }

    def _preprocess(self, img):
        img = self._prepare_for_sobel(img)
        img = self._rescale_values(img)
        img = self._prepare_torch(img)
        return img

    def _preprocess_transformed(self, img):
        img = self._jitter(img)
        img = self._preprocess(img)
        img, _, affine_inverse = self._transformation.apply(img)
        return img, affine_inverse

    def _rescale_values(self, img):
        return img.astype(np.float32) / 255.0

    def _prepare_torch(self, img):
        return torch.from_numpy(img).permute(2, 0, 1).cuda()

    def _prepare_for_sobel(self, img):
        if self._do_sobelize:
            img = custom_greyscale_numpy(img, include_rgb=self._include_rgb)
        return img

    def _pad_and_crop_random(self, img, label):
        required_shape = self._get_required_shape(img)
        start_subscript = get_random_start_subscript(img.shape, required_shape)
        img = reshape_by_pad_crop(img, required_shape, start_subscript)
        required_shape = self._get_required_shape(label)
        label = reshape_by_pad_crop(label, required_shape, start_subscript)
        return img, label

    def _pad_and_crop_center(self, img, label):
        required_shape = self._get_required_shape(img)
        start_subscript = get_center_start_subscript(img.shape, required_shape)
        img = reshape_by_pad_crop(img, required_shape, start_subscript)
        required_shape = self._get_required_shape(label)
        label = reshape_by_pad_crop(label, required_shape, start_subscript)
        return img, label

    def _get_required_shape(self, img):
        required_shape = [*self._input_shape[0:2], *img.shape[2:]]
        return required_shape

    def _jitter(self, img):
        was_gray = False
        if img.shape[-1] == 1:
            was_gray = True
            img = img.squeeze()
        img = Image.fromarray(img.astype(np.uint8))
        img = self._jitter_tf(img)
        img = np.array(img)
        if was_gray:
            img = img[..., np.newaxis]
        return img

    def _sobelize(self, img):
        if img is None:
            return img
        else:
            return sobel_process(img)


class Transformation:
    def __init__(self, config):
        self._use_random_affine = config.transformations.use_random_affine
        self._rotation_range = self._prepare_range(
            config.transformations.rotation_range
        )
        assert self._rotation_range.size == 2
        self._shear_range = self._prepare_range(config.transformations.shear_range)
        assert self._shear_range.size == 2
        self._scale_range = self._prepare_range(config.transformations.scale_range)
        assert self._scale_range.size == 2
        self._flip_horizontal_probability = (
            config.transformations.flip_horizontal_probability
        )

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
