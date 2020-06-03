from pathlib import Path, PurePath
import pickle
from typing import Union

import numpy as np
import yaml

PathLike = Union[str, Path, PurePath]


class CocoFewLabels:
    INCLUSIVE_LOWER_BOUND = -1
    EXCLUSIVE_UPPER_BOUND = 182

    SORTED_COARSE_NAMES = [
        # THINGS
        "electronic-things",  # 0
        "appliance-things",  # 1
        "food-things",  # 2
        "furniture-things",  # 3
        "indoor-things",  # 4
        "kitchen-things",  # 5
        "accessory-things",  # 6
        "animal-things",  # 7
        "outdoor-things",  # 8
        "person-things",  # 9
        "sports-things",  # 10
        "vehicle-things",  # 11
        # STUFF
        "ceiling-stuff",  # 12
        "floor-stuff",  # 13
        "food-stuff",  # 14
        "furniture-stuff",  # 15
        "rawmaterial-stuff",  # 16
        "textile-stuff",  # 17
        "wall-stuff",  # 18
        "window-stuff",  # 19
        "building-stuff",  # 20
        "ground-stuff",  # 21
        "plant-stuff",  # 22
        "sky-stuff",  # 23
        "solid-stuff",  # 24
        "structural-stuff",  # 25
        "water-stuff",  # 26
    ]
    SORTED_COARSE_NAMES_TO_INDEX = {n: i for i, n in enumerate(SORTED_COARSE_NAMES)}

    def __init__(
        self,
        class_count: int,
        use_coarse_labels: bool,
        include_person_things: bool,
        include_animal_things: bool,
        store_path: PathLike,
    ):
        if use_coarse_labels:
            label_names = ["sky-stuff", "plant-stuff", "ground-stuff"]
            if include_person_things:
                label_names.append("person_things")
            if include_animal_things:
                label_names.append("animal_things")
            assert class_count == len(label_names)
            mapper = self._build_mapper(store_path, label_names)
            self._include_person_things = include_person_things
            self._include_animal_things = include_animal_things
            self._fine_to_few_dict = mapper
        self._use_coarse_labels = use_coarse_labels

    def apply(self, label: np.array):
        if self._use_coarse_labels:
            assert self.INCLUSIVE_LOWER_BOUND <= label.min()
            assert label.max() < self.EXCLUSIVE_UPPER_BOUND
            out_label = np.zeros(label.shape, dtype=label.dtype)
            for c in range(self.EXCLUSIVE_UPPER_BOUND):
                out_label[label == c] = self._fine_to_few_dict[c]
        else:
            out_label = label
        return out_label

    @staticmethod
    def _build_mapper(fine_to_coarse_dict_path: PathLike, label_names):
        if not Path(fine_to_coarse_dict_path).exists():
            CocoFewLabels._generate_fine_to_coarse(fine_to_coarse_dict_path)

        with open(fine_to_coarse_dict_path, "rb") as f:
            d = pickle.load(f)
            fine_to_coarse_dict = d["fine_index_to_coarse_index"]

        original_coarse_indices = []
        for label_name in label_names:
            index = CocoFewLabels.SORTED_COARSE_NAMES.index(label_name)
            original_coarse_indices.append(index)

        fine_to_few_dict = {}
        for fine_index in range(CocoFewLabels.EXCLUSIVE_UPPER_BOUND):
            coarse_index = fine_to_coarse_dict[fine_index]
            if coarse_index in original_coarse_indices:
                few_index = original_coarse_indices.index(coarse_index)
            else:
                few_index = -1
            fine_to_few_dict[fine_index] = few_index

        return fine_to_few_dict

    @staticmethod
    def _generate_fine_to_coarse(out_path: str):
        with open("cocostuff_fine_raw.txt") as f:
            l = [tuple(pair.rstrip().split("\t")) for pair in f]
            l = [(int(ind), name) for ind, name in l]

        with open("cocostuff_hierarchy.yml") as f:
            d = yaml.load(f)

        fine_index_to_coarse_index = {}
        fine_name_to_coarse_name = {}
        for fine_ind, fine_name in l:
            assert fine_ind >= 0 and fine_ind < 182
            parent_name = list(CocoFewLabels._find_parent(fine_name, d))
            assert len(parent_name) == 1
            parent_name = parent_name[0]
            parent_ind = CocoFewLabels.SORTED_COARSE_NAMES_TO_INDEX[parent_name]
            assert parent_ind >= 0 and parent_ind < 27

            fine_index_to_coarse_index[fine_ind] = parent_ind
            fine_name_to_coarse_name[fine_name] = parent_name

        assert len(fine_index_to_coarse_index) == 182

        with open(out_path, "wb") as out_f:
            pickle.dump(
                {
                    "fine_index_to_coarse_index": fine_index_to_coarse_index,
                    "fine_name_to_coarse_name": fine_name_to_coarse_name,
                },
                out_f,
            )

        with open(out_path + ".txt", "w") as out_f:
            for k, v in fine_name_to_coarse_name.items():
                out_f.write("%s\t%s" % (k, v))

            for k, v in fine_index_to_coarse_index.items():
                out_f.write("%d\t%d" % (k, v))

    @staticmethod
    def _find_parent(name, d):
        for k, v in d.items():
            if isinstance(v, list):
                if name in v:
                    yield k
            else:
                assert isinstance(v, dict)
                for res in _find_parent(name, v):
                    yield res
