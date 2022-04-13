from pathlib import Path, PurePath
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

import inc.python_image_utilities.image_util as iutil
import preprocessing as pre
import utils

PathLike = Union[str, Path, PurePath]


class ImageFolderDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        image_folder: PathLike,
        image_info: utils.ImageInfo,
        preprocessor: pre.ImagePreprocessor,
        extensions: List[str],
        label_folder: Optional[PathLike] = None,
    ):
        assert 1 <= len(extensions)
        image_files = self._build_files(image_folder, extensions)
        assert 1 <= len(image_files)
        if label_folder is not None:
            label_files = self._build_files(label_folder, extensions)
            assert 1 <= len(label_files)
            assert len(label_files) == len(image_files)
            assert self._check_labels(image_files, label_files)
        else:
            label_files = None
        self._pre = preprocessor
        self._image_info = image_info
        self._image_files = image_files
        self._label_files = label_files

    def __len__(self):
        return len(self._image_files)

    def __getitem__(self, index):
        data = self._load_data(index)
        out = self._pre.apply(**data)
        for k, v in out.items():
            assert k is not None
            assert v is not None
        return out

    def _load_data(self, index):
        image_file_path = self._image_files[index]
        image = self._load_image(image_file_path)
        data = {"image": image, "file_path": str(image_file_path)}
        if self._label_files is not None:
            label_file_path = self._label_files[index]
            label = self._load_label(label_file_path)
            assert label.ndim == 2
            assert label.shape[:2] == image.shape[:2]
            data["label"] = label
        return data

    def _load_image(self, file_path):
        force_rgb = self._image_info.is_rgb
        image = iutil.load(path=file_path, force_rgb=force_rgb)
        assert image.ndim == 3
        image = image.astype(np.uint8)
        if not force_rgb and image.ndim == 3:
            image = image.squeeze()
            assert image.ndim == 2
        return image

    def _load_label(self, file_path: PathLike):
        image = iutil.load(path=file_path, force_rgb=False)
        if image.ndim == 3:
            image = image.squeeze()
        assert image.ndim == 2
        label = image.astype(np.int32)
        label[label == 255] = -1
        return label

    @staticmethod
    def _build_files(folder: PathLike, extensions: List[str]):
        files = []
        for extension in extensions:
            glob = "**/*{:s}".format(extension)
            files.extend(list(Path(folder).glob(glob)))
        files = sorted(files, key=ImageFolderDataset._get_file_key)
        return files

    @staticmethod
    def _check_labels(train_files: List[PathLike], label_files: List[PathLike]):
        matches = True
        for train, label in zip(train_files, label_files):
            if PurePath(train).stem != PurePath(label).stem:
                matches = False
                break
        return matches

    @staticmethod
    def _get_file_key(file_path: PathLike):
        path = PurePath(file_path)
        parents = list(reversed(path.parents))
        return tuple([path.stem, *parents])


class TrainDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset: ImageFolderDataset, batch_size: int, shuffle: bool):
        super(TrainDataLoader, self).__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            drop_last=False,
        )


class TestDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset: ImageFolderDataset, batch_size: int):
        super(TestDataLoader, self).__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )


class EvalDataset(ImageFolderDataset):
    def __init__(
        self,
        eval_folder: PathLike,
        input_size: int,
        image_info: utils.ImageInfo,
        preprocessor: pre.EvalImagePreprocessor,
        extensions: List[str] = [".png"],
        batch_size: int = 128,
    ):
        super(EvalDataset, self).__init__(
            image_folder=eval_folder,
            image_info=image_info,
            preprocessor=preprocessor,
            extensions=extensions,
        )
        self._input_size = input_size
        self._batch_size = batch_size

    def __len__(self):
        return len(self._image_files)

    def __getitem__(self, index):
        """
        Returns a dict with the following:
            1. "image": The original image content
            2. "batches": A list of batches as dicts expected by the arch
            3. "patch_count": Number of total patches in the overall image, used
               by reassemble
            4. "padding": Used by reassemble
            5. "file_path": Path to the original file
        """
        out = self._load_data(index)
        if out["image"].ndim == 2:
            out["image"] = out["image"][..., np.newaxis]
        image_shape = out["image"].shape
        patches = iutil.patchify_image(
            out["image"], patch_shape=(self._input_size, self._input_size)
        )
        patches = patches.squeeze()
        patches = [self._pre.apply(image=p)["image"] for p in patches]
        batches = [
            patches[i : i + self._batch_size]
            for i in range(0, len(patches), self._batch_size)
        ]
        batches = [torch.stack(batch, dim=0) for batch in batches]
        batches = [{"image": batch} for batch in batches]
        out.update({"batches": batches, "image_shape": image_shape})

        for k, v in out.items():
            assert k is not None
            assert v is not None
        return out

    def _batches(self, patches):
        for index in range(0, len(patches), self._batch_size):
            yield patches[index : index + self._batch_size]

    @staticmethod
    def reassemble(image: np.ndarray, image_shape):
        return iutil.unpatchify_image(
            patches=image, image_shape=image_shape, offset=(0, 0)
        )


class EvalDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset: EvalDataset):
        super(EvalDataLoader, self).__init__(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            collate_fn=self._collate,
        )

    def reassemble(self, image: np.ndarray, image_shape):
        return self.dataset.reassemble(image, image_shape)

    @staticmethod
    def _collate(data: List[Dict[str, Any]]) -> Dict[str, Any]:
        assert len(data) == 1
        return data[0]
