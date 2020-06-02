import matplotlib
import numpy as np
import torch
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os


def render(data, mode, name, colour_map=None, offset=0, out_dir="", labels=None):
    assert mode in ("label", "mask", "matrix", "preds") or "image" in mode

    if isinstance(data, torch.Tensor):
        if data.is_cuda:
            data = data.cpu()
        data = data.numpy()

    has_channels = "image" in mode or mode == "label"

    do_loop = False
    if has_channels and data.ndim == 4:
        do_loop = True
    elif has_channels and data.ndim == 2:
        data = data[..., np.newaxis]
    elif not has_channels and data.ndim == 3:
        do_loop = True

    if do_loop:
        for i in range(data.shape[0]):
            img = data[i, ...]
            ind_name = name + "_{:d}".format(i)
            render(img, mode=mode, name=ind_name, out_dir=out_dir)
        return

    if has_channels:
        assert data.ndim == 3
    else:
        assert data.ndim == 2

    # recursively called case for single inputs
    out_handle = os.path.join(out_dir, name)

    if mode == "image":
        RGB = "rgb"
        GRAY = "gray"
        SOBEL_H = "sobel_h"
        SOBEL_V = "sobel_v"
        data = data.transpose((1, 2, 0))  # channels last
        imgs = {RGB: None, GRAY: None, SOBEL_H: None, SOBEL_V: None}
        if data.shape[2] == 1:
            imgs[GRAY] = data[:, :, 0]
        elif data.shape[2] == 3:
            imgs[RGB] = data
        elif data.shape[2] == 4:
            imgs[RGB] = data[:, :, :3]
            imgs[GRAY] = data[:, :, -1]
        elif data.shape[2] == 5:
            imgs[RGB] = data[:, :, :3]
            imgs[SOBEL_H] = data[:, :, 3]
            imgs[SOBEL_V] = data[:, :, 4]
        for key, img in imgs.items():
            if img is None:
                continue
            img *= 255.0
            out = Image.fromarray(img.astype(np.uint8))
            out.save("_".join([out_handle, key]) + ".png")

    elif mode == "image_ir":
        data = data.transpose((1, 2, 0))  # channels last
        if data.shape[2] == 5:  # rgb, grey, ir
            # pre-sobel with rgb
            # don't render grey, only colour
            data = data[:, :, :3]
        elif data.shape[2] == 2:  # grey, ir
            # pre-sobel no rgb
            # render grey
            data = data[:, :, 0]
        elif (data.shape[2] == 3) or (data.shape[2] == 4):  # no sobel
            data = data[:, :, :3]

        data *= 255.0
        img = Image.fromarray(data.astype(np.uint8))
        img.save(out_handle + ".png")

    elif mode == "image_as_feat":
        data = data.transpose((1, 2, 0))
        if data.shape[2] == 5:
            # post-sobel with rgb

            # only render sobel

            data_sobel = data[:, :, [3, 4]].sum(axis=2, keepdims=False) * 0.5 * 255.0
            img_sobel = Image.fromarray(data_sobel.astype(np.uint8))
            img_sobel.save(out_handle + ".png")
            return

            data = data[:, :, :3]
        elif data.shape[2] == 2:
            # post_sobel no rgb

            # only render sobel

            data = data.sum(axis=2, keepdims=False) * 0.5
        else:
            assert False

        data *= 255.0
        img = Image.fromarray(data.astype(np.uint8))
        img.save(out_handle + ".png")

    elif mode == "mask":
        # only has 1s and 0s, whatever the dtype
        img = Image.fromarray(data.astype(np.uint8) * 255)
        img.save(out_handle + ".png")

    elif mode == "label":
        # render histogram, with title (if current labels contains 0-11, 12-26,
        # 0-91, 92-181)
        assert data.dtype == np.int32 or data.dtype == np.int64
        if labels is None:
            labels = data

        # 0 (-1), [1 (0), 12 (11)], [13 (12), 27 (26)]
        hist = _make_hist(data, labels)
        inds = np.nonzero(hist > 0)[0]
        min_ind = inds.min()
        max_ind = inds.max()

        fig, ax = plt.subplots(1, figsize=(20, 20))
        ax.plot(hist)
        ax.set_title("Labels for %s, min %s, max %s" % (name, min_ind, max_ind))
        fig.canvas.draw_idle()
        fig.savefig(out_handle + ".png")
        plt.close(fig)

    elif mode == "matrix":
        with open(out_handle + ".txt", "w") as f:
            f.write(str(data))

    elif mode == "preds":
        h, w = data.shape
        img = np.zeros((h, w, 3), dtype=np.uint8)
        # ignore <0 labels
        for c in range(0, data.max() + 1):
            img[data == c, :] = colour_map[c]

        img = Image.fromarray(img)
        img.save(out_handle + ".png")

    else:
        assert False


def _make_hist(tensor, values):
    res = np.zeros(len(values))
    for i, v in enumerate(values):
        res[i] = (tensor == v).sum()
    return res
