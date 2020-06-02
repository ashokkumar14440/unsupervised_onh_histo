from typing import List

import torch
import torch.nn.functional as F

STRUCTURE = [(64, 1), (128, 1), ("M", None), (256, 1), (256, 1), (512, 2), (512, 2)]


class VGGTrunk(torch.nn.Module):
    def __init__(
        self,
        structure: List[tuple],
        input_channels: int,
        convolution_size: int,
        padding: int,
        stride: int,
        batch_norm: bool = True,
        batch_norm_tracking: bool = True,
    ):
        super(VGGTrunk, self).__init__()

        layers = []
        current_channels = input_channels
        for next_channels, dilation in structure:
            if next_channels == "M":
                new_layers = [torch.nn.MaxPool2d(kernel_size=2, stride=2)]
            elif next_channels == "A":
                new_layers = [torch.nn.AvgPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = torch.nn.Conv2d(
                    current_channels,
                    next_channels,
                    kernel_size=convolution_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
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
        self._input_channels = input_channels
        self._convolution_size = convolution_size
        self._padding = padding
        self._stride = stride
        self._batch_norm = batch_norm
        self._batch_norm_tracking = batch_norm_tracking
        self._layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self._layers(x)  # do not flatten
        return x


class SegmentationNet10aHead(torch.nn.Module):
    def __init__(
        self, feature_count: int, input_size: int, class_count: int, subhead_count: int
    ):
        assert input_size % 2 == 0
        super(SegmentationNet10aHead, self).__init__()
        self._heads = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Conv2d(
                        feature_count,
                        class_count,
                        kernel_size=1,
                        stride=1,
                        dilation=1,
                        padding=1,
                        bias=False,
                    ),
                    torch.nn.Softmax2d(),
                )
                for _ in range(subhead_count)
            ]
        )
        self._input_size = input_size

    def forward(self, x):
        results = []
        for head in self._heads:
            x_head = head(x)
            x_head = F.interpolate(
                x_head, size=self._input_size, mode="bilinear", align_corners=True
            )
            results.append(x_head)
        return results


class SegmentationNet10aTwoHead(torch.nn.Module):
    def __init__(
        self,
        trunk: VGGTrunk,
        head_A: SegmentationNet10aHead,
        head_B: SegmentationNet10aHead,
    ):
        super(SegmentationNet10aTwoHead, self).__init__()
        self._trunk = trunk
        self._head_A = head_A
        self._head_B = head_B

        self._initialize_weights()

    def forward(self, x, head="B"):
        x = self._trunk(x)
        if head == "A":
            x = self._head_A(x)
        elif head == "B":
            x = self._head_B(x)
        else:
            assert False
        return x

    def _initialize_weights(self, mode="fan_in"):
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
