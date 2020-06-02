from pathlib import Path, PurePath

import cv2
import numpy as np
import torch

from src.scripts.segmentation.preprocess import Preprocessor
from src.utils.segmentation.render import render


class OnhDataset(torch.utils.data.Dataset):
    def __init__(self, config, split, purpose, preprocessor: Preprocessor):
        self._purpose = purpose
        self._split = split
        self._root = PurePath(config.dataset.root)
        self._do_render = config.output.rendering.enabled
        self._render_folder = (
            PurePath(config.out_dir) / config.output.rendering.output_subfolder
        )
        Path(self._render_folder).mkdir(parents=True, exist_ok=True)

        assert self._is_gt_k_ok(config.architecture.head_B_class_count)
        self._gt_k = config.architecture.head_B_class_count
        self._input_size = config.dataset.parameters.input_size
        self._preprocessor = preprocessor

        self._files = []

        cv2.setNumThreads(0)

        self._set_files()

    def _prepare_train(self, index, img, label):
        assert img.shape[:2] == label.shape

        # PREPROCESS
        self._preprocessor.purpose = self._purpose  # ! probably race condition
        result = self._preprocessor.apply(img, label)
        img = result["image"]
        label = result["label"]
        affine_inverse = result["affine_inverse"]
        t_img = result["transformed_image"]

        # CREATE AND USE MASK
        _, mask = self._generate_mask(label)
        img = self._apply_mask(img, mask)
        t_img = self._apply_mask(t_img, mask)

        # RENDER
        if self._do_render:
            self._render(img, mode="image", name="train_data_img_{:d}".format(index))
            self._render(
                t_img, mode="image", name="train_data_t_img_{:d}".format(index)
            )
            self._render(
                affine_inverse,
                mode="matrix",
                name="train_data_affine_inverse_{:d}".format(index),
            )
            self._render(mask, mode="mask", name="train_data_mask_{:d}".format(index))

        return img, t_img, affine_inverse, mask

    def _prepare_test(self, index, img, label):
        assert img.shape[:2] == label.shape

        # PREPROCESS
        self._preprocessor.purpose = self._purpose  # ! probable race condition
        result = self._preprocessor.apply(img, label)
        img = result["image"]
        label = result["label"]

        # RENDER BEFORE
        if self._do_render:
            self._render(
                label,
                mode="label",
                name="test_data_label_{:d}".format(index),
                labels=list(range(self._gt_k)),
            )

        # CREATE AND USE MASK
        label, mask = self._generate_mask(label)
        img = self._apply_mask(img, mask)

        # RENDER
        if self._do_render:
            self._render(img, mode="image", name="test_data_img_{:d}".format(index))
            self._render(
                label,
                mode="label",
                name="test_data_label_post_{:d}".format(index),
                labels=list(range(self._gt_k)),
            )
            self._render(mask, mode="mask", name="test_data_mask_{:d}".format(index))

        return img, label, mask

    def __getitem__(self, index):
        file_path = self._files[index]
        image, label = self._load_data(file_path)

        if self._purpose == "train":
            return self._prepare_train(index, image, label)
        elif self._purpose == "test":
            return self._prepare_test(index, image, label)
        else:
            assert False

    def __len__(self):
        return len(self._files)

    def _generate_mask(self, label):
        label_out, mask = self._filter_label(label)
        assert mask.dtype == np.bool
        mask = torch.from_numpy(mask.astype(np.uint8)).cuda()
        return label_out, mask

    def _render(self, img, mode, name, **kwargs):
        render(img, mode=mode, name=name, out_dir=self._render_folder, **kwargs)

    def _is_gt_k_ok(self, supplied_gt_k):
        # VIRTUAL
        return True

    def _filter_label(self, label):
        # VIRTUAL
        return label, label >= 0

    def _set_files(self):
        # VIRTUAL
        self._files = list(Path(self._root / self._split).glob("*.png"))

    def _load_data(self, image_path):
        # VIRTUAL
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE).astype(np.uint8)
        label = image > 0
        return image, label

    def _apply_mask(self, img, mask):
        return img
