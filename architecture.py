from pathlib import Path, PurePath
import pickle
from typing import Any, Dict, List, Union

import torch
import torch.nn.functional as F


PathLike = Union[Path, PurePath, str]


class Structure:
    CHANNELS = "channels"
    DILATION = "dilation"

    def __init__(self, input_channels: int, structure: List[list]):
        assert 0 < input_channels
        for s in structure:
            assert len(s) == 2
            if isinstance(s[0], int):
                assert isinstance(s[1], int)
            elif isinstance(s[0], str):
                assert s[0] in ("M", "A")
                assert s[1] is None
            else:
                assert False

        self._structure = structure
        self._input_channels = input_channels

    @property
    def input_channels(self):
        return self._input_channels

    @property
    def feature_count(self):
        return self[-1][self.CHANNELS]

    def __len__(self):
        return len(self._structure)

    def __getitem__(self, index):
        s = self._structure[index]
        return {self.CHANNELS: s[0], self.DILATION: s[1]}

    def __iter__(self):
        return (self[i] for i in range(len(self)))


class VGGTrunk(torch.nn.Module):
    _POOL_PARAMETERS = {"kernel_size": 2, "stride": 2}

    def __init__(
        self,
        structure: Structure,
        convolution_size: int,
        padding: int,
        stride: int,
        batch_norm: bool,
        batch_norm_tracking: bool,
    ):
        super(VGGTrunk, self).__init__()

        layers = []
        current_channels = structure.input_channels
        for s in structure:
            next_channels = s[Structure.CHANNELS]
            if next_channels == "M":
                new_layers = [torch.nn.MaxPool2d(**self._POOL_PARAMETERS)]
            elif next_channels == "A":
                new_layers = [torch.nn.AvgPool2d(**self._POOL_PARAMETERS)]
            else:
                assert isinstance(next_channels, int)
                conv2d = torch.nn.Conv2d(
                    current_channels,
                    next_channels,
                    kernel_size=convolution_size,
                    stride=stride,
                    padding=padding,
                    dilation=s[Structure.DILATION],
                    bias=False,
                )
                if batch_norm:
                    new_layers = [
                        conv2d,
                        torch.nn.BatchNorm2d(
                            next_channels, track_running_stats=batch_norm_tracking
                        ),
                        torch.nn.ReLU(inplace=True),
                    ]
                else:
                    new_layers = [conv2d, torch.nn.ReLU(inplace=True)]
                current_channels = next_channels
            layers.extend(new_layers)

        self._structure = structure
        self._layers = torch.nn.Sequential(*layers)

    @property
    def feature_count(self):
        return self._structure.feature_count

    def forward(self, x):
        x = self._layers(x)  # do not flatten
        return x


class SegmentationNet10aHead(torch.nn.Module):
    _CONV_PARAMETERS = {
        "kernel_size": 1,
        "stride": 1,
        "dilation": 1,
        "padding": 1,
        "bias": False,
    }

    def __init__(
        self, feature_count: int, input_size: int, class_count: int, subhead_count: int
    ):
        assert input_size % 2 == 0
        super(SegmentationNet10aHead, self).__init__()
        subheads = []
        for _ in range(subhead_count):
            conv = torch.nn.Conv2d(feature_count, class_count, **self._CONV_PARAMETERS)
            softmax = torch.nn.Softmax2d()
            head = torch.nn.Sequential(conv, softmax)
            subheads.append(head)
        subheads = torch.nn.ModuleList(subheads)
        self._subheads = subheads
        self._input_size = input_size

    def forward(self, x):
        results = []
        for subhead in self._subheads:
            x_head = subhead(x)
            x_head = F.interpolate(
                x_head, size=self._input_size, mode="bilinear", align_corners=True
            )
            results.append(x_head)
        return results


class HeadsInfo:
    def __init__(
        self, heads_info: List[Dict[str, Any]], input_size: int, subhead_count: int
    ):
        assert 0 < subhead_count
        assert 0 < input_size

        heads = {}
        primary_head = None
        class_count = None
        for info in heads_info:
            label = info["label"]
            assert label not in heads
            heads[label] = info["class_count"]
            if "primary" in info:
                assert primary_head is None
                assert class_count is None
                primary_head = label
                class_count = info["class_count"]
        assert primary_head is not None
        assert class_count is not None

        for label, count in heads.items():
            if label != primary_head:
                assert count >= class_count

        self._subhead_count = subhead_count
        self._input_size = input_size
        self._heads = heads
        self._primary_head = primary_head
        self._class_count = class_count

    @property
    def subhead_count(self):
        return self._subhead_count

    @property
    def primary_head(self):
        return self._primary_head

    @property
    def class_count(self):
        return self._class_count

    @property
    def order(self):
        return list(self._heads.keys())

    def build_heads(self, feature_count: int) -> Dict[str, SegmentationNet10aHead]:
        heads = {}
        for label, class_count in self._heads.items():
            assert label not in heads
            heads[label] = SegmentationNet10aHead(
                feature_count=feature_count,
                input_size=self._input_size,
                class_count=class_count,
                subhead_count=self._subhead_count,
            )
        return heads

    def save(self, file_path: PathLike) -> None:
        with open(file_path, mode="wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file_path: PathLike):
        with open(file_path, mode="rb") as f:
            out = pickle.load(f)
        return out


class SegmentationNet10aTwoHead(torch.nn.Module):
    def __init__(self, trunk: VGGTrunk, heads: Dict[str, SegmentationNet10aHead]):
        super(SegmentationNet10aTwoHead, self).__init__()

        self._trunk = trunk
        self._heads = torch.nn.ModuleDict(heads)
        self._initialize_weights()

    def forward(self, data, head):
        images = self._forward_image(data["image"], head=head)
        t_images = self._forward_image(data["transformed_image"], head=head)
        count = len(images)
        out = {"count": count, "image": images, "transformed_image": t_images}
        for k, v in data.items():
            if k not in out:
                out[k] = v
        return out

    def _forward_image(self, x, head):
        assert head in self._heads
        x = self._trunk(x)
        x = self._heads[head](x)
        return x

    def _initialize_weights(self, mode="fan_in"):
        # TODO heads are not part of
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode=mode, nonlinearity="relu")
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, torch.nn.BatchNorm2d) or isinstance(
                m, torch.nn.BatchNorm1d
            ):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, torch.nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
