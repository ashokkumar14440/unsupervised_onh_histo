import functools
from pathlib import Path, PurePath
import pickle
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

import utils


PathLike = Union[Path, PurePath, str]


class Loss:
    def __init__(self, heads: List[str]):
        self._heads = heads

    def __call__(
        self, head: str, data: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros((1, 1)), torch.zeros((1, 1))

    def save(self, file_path: PathLike):
        with open(file_path, mode="wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file_path: PathLike):
        with open(file_path, mode="rb") as f:
            out = pickle.load(f)
        return out


class IIDLoss(Loss):
    EPSILON = sys.float_info.epsilon

    def __init__(
        self,
        heads: List[str],
        lambs: Dict[str, float],
        use_uncollapsed: bool,
        half_T_side_dense: int,
        half_T_side_sparse_min: int,
        half_T_side_sparse_max: int,
        output_files: Optional[utils.OutputFiles] = None,
        do_render: Optional[bool] = None,
    ):
        super(IIDLoss, self).__init__(heads=heads)
        for head in heads:
            if head not in lambs:
                lambs[head] = 1.0

        if do_render is None:
            do_render = False
        if do_render:
            assert output_files is not None

        self._lambs = lambs
        self._use_uncollapsed = use_uncollapsed
        self._hts_dense = half_T_side_dense
        self._hts_sparse_min = half_T_side_sparse_min
        self._hts_sparse_max = half_T_side_sparse_max
        self._output_files = output_files
        self._do_render = do_render

    def __call__(
        self, head: str, data: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert head in self._lambs
        assert "count" in data
        count = data["count"]
        assert 0 < count

        lamb = self._lambs[head]
        losses = []
        losses_no_lamb = []
        for i in range(count):
            loss, loss_no_lamb = self._loss(
                lamb=lamb,
                file_path=data["file_path"],
                x1=data["image"][i],
                x2=data["transformed_image"][i],
                affine_inverse=data.get("affine_inverse", None),
                mask=data.get("mask", None),
            )
            losses.append(loss)
            losses_no_lamb.append(loss_no_lamb)
        avg_loss = functools.reduce(lambda x, y: x + y, losses) / count
        avg_loss_no_lamb = functools.reduce(lambda x, y: x + y, losses) / count
        assert avg_loss.requires_grad
        return avg_loss, avg_loss_no_lamb

    def _loss(
        self,
        lamb: float,
        file_path: List[PathLike],
        x1: torch.Tensor,
        x2: torch.Tensor,
        affine_inverse: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert x1.requires_grad
        assert x2.requires_grad
        assert x1.shape == x2.shape

        bn, k, h, w = x1.shape

        x2_in = x2
        if affine_inverse is not None:
            assert not affine_inverse.requires_grad
            x2 = _perform_affine_tf(x2, affine_inverse)
        if mask is not None:
            assert not mask.requires_grad
        else:
            mask = torch.ones((bn, h, w)).to(device=x1.device)

        # TODO MASK IS WRONG SHAPE, should be n, 1, h, w
        # TODO may be error in undoing affine T

        if (self._hts_sparse_min != 0) or (self._hts_sparse_max != 0):
            x2 = _random_translation_multiple(
                x2,
                half_side_min=self._hts_sparse_min,
                half_side_max=self._hts_sparse_max,
            )

        if self._do_render:
            name = PurePath(file_path[0]).stem
            self._output_files.save_confidence_tensor(name + "_loss_labels", x1[0])
            self._output_files.save_confidence_tensor(
                name + "_loss_transf_labels", x2_in[0]
            )
            self._output_files.save_confidence_tensor(
                name + "_loss_untransf_labels", x2[0]
            )
            self._output_files.save_mask_tensor(name + "_loss_mask", mask[0])

        # zero out all irrelevant patches
        mask = mask.view(bn, 1, h, w)  # mult, already float32
        x1 = x1 * mask  # broadcasts
        x2 = x2 * mask

        # sum over everything except classes, by convolving x1_outs with x2_outs_inv
        # which is symmetric, so doesn't matter which one is the filter
        x1 = x1.permute(1, 0, 2, 3).contiguous()  # k, ni, h, w
        x2 = x2.permute(1, 0, 2, 3).contiguous()  # k, ni, h, w

        # (k, k, T, T)
        p_i_j = F.conv2d(x1, weight=x2, padding=(self._hts_dense, self._hts_dense))

        if self._use_uncollapsed:
            (p_i_j, p_i, p_j, count) = self._uncollapsed(p_i_j, k)
        else:
            (p_i_j, p_i, p_j, count) = self._collapsed(p_i_j)

        # log-stability
        p_i_j[(p_i_j < self.EPSILON).data] = self.EPSILON
        p_i[(p_i < self.EPSILON).data] = self.EPSILON
        p_j[(p_j < self.EPSILON).data] = self.EPSILON

        # maximise information
        loss = (
            -p_i_j * (torch.log(p_i_j) - lamb * torch.log(p_i) - lamb * torch.log(p_j))
        ).sum() / count

        # for analysis only
        loss_no_lamb = (
            -p_i_j * (torch.log(p_i_j) - torch.log(p_i) - torch.log(p_j))
        ).sum() / count

        return loss, loss_no_lamb

    def _uncollapsed(self, p_i_j, k):
        # p_i_j -> (k, k, T, T)
        T = self._hts_dense * 2 + 1

        p_i_j = p_i_j.permute(2, 3, 0, 1)
        p_i_j = p_i_j / p_i_j.sum(dim=3, keepdim=True).sum(dim=2, keepdim=True)  # norm

        # symmetrise, transpose the k x k part
        p_i_j = (p_i_j + p_i_j.permute(0, 1, 3, 2)) / 2.0

        # T x T x k x k
        p_i = p_i_j.sum(dim=2, keepdim=True).repeat(1, 1, k, 1)
        p_j = p_i_j.sum(dim=3, keepdim=True).repeat(1, 1, 1, k)

        # pij -> (T, T, k, k)
        # pi -> (1, 1, k, 1)
        # pj -> (1, 1, 1, k)
        return (p_i_j, p_i, p_j, T)

    def _collapsed(self, p_i_j):
        # p_i_j -> (k, k, T, T)
        p_i_j = p_i_j.sum(dim=2, keepdim=False).sum(dim=2, keepdim=False)  # k, k

        # normalise, use sum, not bn * h * w * T_side * T_side, because we use a mask
        # also, some pixels did not have a completely unmasked box neighbourhood,
        # but it's fine - just less samples from that pixel
        current_norm = float(p_i_j.sum())
        p_i_j = p_i_j / current_norm

        # symmetrise
        p_i_j = (p_i_j + p_i_j.t()) / 2.0

        # compute marginals
        p_i = p_i_j.sum(dim=1).unsqueeze(1)  # k, 1
        p_j = p_i_j.sum(dim=0).unsqueeze(0)  # 1, k

        # pij -> (k, k)
        # pi -> (k, 1)
        # pj -> (1, k)
        return (p_i_j, p_i, p_j, 1)


def _perform_affine_tf(data, inverse):
    assert data.shape[0] == inverse.shape[0]
    assert inverse.shape[1] == 2
    assert inverse.shape[2] == 3

    grid = F.affine_grid(
        inverse, data.shape, align_corners=True  # type: ignore
    )  # output should be same size
    return F.grid_sample(
        data, grid, padding_mode="zeros", align_corners=True  # type: ignore
    )  # this can ONLY do bilinear


def _random_translation_multiple(data, half_side_min: int, half_side_max: int):
    h = data.shape[2]
    w = data.shape[3]

    # pad last 2, i.e. spatial, dimensions, equally in all directions
    data = F.pad(
        data,
        [half_side_max, half_side_max, half_side_max, half_side_max],
        "constant",
        0,
    )
    assert data.shape[2:] == (2 * half_side_max + h, 2 * half_side_max + w)

    # random x, y displacement
    t = np.random.randint(half_side_min, half_side_max + 1, size=(2,))
    polarities = np.random.choice([-1, 1], size=(2,), replace=True)
    t *= polarities

    # -x, -y in orig img frame is now -x+half_side_max, -y+half_side_max in new
    t += half_side_max

    data = data[:, :, t[1] : (t[1] + h), t[0] : (t[0] + w)]
    assert data.shape[2:] == (h, w)

    return data
