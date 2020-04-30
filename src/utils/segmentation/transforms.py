import cv2
import numpy as np
import torch
import torch.nn.functional as F


def custom_greyscale_numpy(img, include_rgb=True):
    # Takes and returns a channel-last numpy array, uint8

    # use channels last for cvtColor
    h, w, c = img.shape
    grey_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).reshape(h, w, 1)  # new memory

    if include_rgb:
        img = np.concatenate([img, grey_img], axis=2)
    else:
        img = grey_img

    return img


def get_random_start_subscript(shape, required_shape):
    shape_change = _calculate_crop_change(shape, required_shape)
    return np.array([np.random.randint(low=c, high=1) for c in shape_change])


def get_center_start_subscript(shape, required_shape):
    shape_change = _calculate_crop_change(shape, required_shape)
    return shape_change // 2


def reshape_by_pad_crop(data, required_shape, crop_start_subscript=None):
    """
    Padding occurs along a dimension if the data array is smaller than the
    required shape. The start and end of a dimension are padded equally, with
    the end receiving one extra for odd differences. Cropping occurs if the data
    is larger than required. The start subscript depicts the starting point of
    the cropped data in the input data.
    """
    assert data.ndim == len(required_shape)
    out = pad(data, required_shape)
    if crop_start_subscript is None:
        crop_start_subscript = get_center_start_subscript(data.shape, required_shape)
    out = crop(out, required_shape, crop_start_subscript)
    return out


def pad(data, required_shape):
    change = _calculate_pad_change(data.shape, required_shape)
    pre_change = change // 2
    post_change = change - pre_change
    padding = list(zip(pre_change, post_change))  # post pad only
    return np.pad(data, padding)


def crop(data, required_shape, start_subscript):
    change = _calculate_crop_change(data.shape, required_shape)
    pre_change = np.array(list(start_subscript))
    post_change = change - pre_change
    start = -pre_change
    end = np.array(list(data.shape)) + post_change
    sl = tuple(slice(s, e) for s, e in zip(start, end))
    return data[sl]


def _calculate_pad_change(shape, required_shape):
    return _calculate_shape_change(shape, required_shape, limit_fn=lambda x: max(0, x))


def _calculate_crop_change(shape, required_shape):
    return _calculate_shape_change(shape, required_shape, limit_fn=lambda x: min(0, x))


def _calculate_shape_change(shape, required_shape, limit_fn=None):
    """Positive means pad, negative means crop."""
    assert len(shape) == len(required_shape)
    shape = np.array(list(shape))
    req = np.array(list(required_shape))
    change = req - shape
    change = np.array([limit_fn(c) for c in change])
    return change


def random_affine(
    img,
    min_rot=None,
    max_rot=None,
    min_shear=None,
    max_shear=None,
    min_scale=None,
    max_scale=None,
):
    # Takes and returns torch cuda tensors with channels 1st (1 img)
    # rot and shear params are in degrees
    # tf matrices need to be float32, returned as tensors
    # we don't do translations

    # https://github.com/pytorch/pytorch/issues/12362
    # https://stackoverflow.com/questions/42489310/matrix-inversion-3-3-python
    # -hard-coded-vs-numpy-linalg-inv

    # https://github.com/pytorch/vision/blob/master/torchvision/transforms
    # /functional.py#L623
    # RSS(a, scale, shear) = [cos(a) *scale   - sin(a + shear) * scale     0]
    #                        [ sin(a)*scale    cos(a + shear)*scale     0]
    #                        [     0                  0          1]
    # used by opencv functional _get_affine_matrix and
    # skimage.transform.AffineTransform

    assert len(img.shape) == 3
    a = np.radians(np.random.rand() * (max_rot - min_rot) + min_rot)
    shear = np.radians(np.random.rand() * (max_shear - min_shear) + min_shear)
    scale = np.random.rand() * (max_scale - min_scale) + min_scale

    affine1_to_2 = np.array(
        [
            [np.cos(a) * scale, -np.sin(a + shear) * scale, 0.0],
            [np.sin(a) * scale, np.cos(a + shear) * scale, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )  # 3x3

    affine2_to_1 = np.linalg.inv(affine1_to_2).astype(np.float32)

    affine1_to_2, affine2_to_1 = affine1_to_2[:2, :], affine2_to_1[:2, :]  # 2x3
    affine1_to_2, affine2_to_1 = (
        torch.from_numpy(affine1_to_2).cuda(),
        torch.from_numpy(affine2_to_1).cuda(),
    )

    img = perform_affine_tf(img.unsqueeze(dim=0), affine1_to_2.unsqueeze(dim=0))
    img = img.squeeze(dim=0)

    return img, affine1_to_2, affine2_to_1


def perform_affine_tf(data, tf_matrices):
    # expects 4D tensor, we preserve gradients if there are any

    n_i, k, h, w = data.shape
    n_i2, r, c = tf_matrices.shape
    assert n_i == n_i2
    assert r == 2 and c == 3

    grid = F.affine_grid(
        tf_matrices, data.shape, align_corners=True  # type: ignore
    )  # output should be same size
    data_tf = F.grid_sample(
        data, grid, padding_mode="zeros", align_corners=True  # type: ignore
    )  # this can ONLY do bilinear

    return data_tf


def random_translation_multiple(data, half_side_min, half_side_max):
    n, c, h, w = data.shape

    # pad last 2, i.e. spatial, dimensions, equally in all directions
    data = F.pad(
        data,
        (half_side_max, half_side_max, half_side_max, half_side_max),
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


def random_translation(img, half_side_min, half_side_max):
    # expects 3d (cuda) tensor with channels first
    c, h, w = img.shape

    # pad last 2, i.e. spatial, dimensions, equally in all directions
    img = F.pad(
        img, (half_side_max, half_side_max, half_side_max, half_side_max), "constant", 0
    )
    assert img.shape[1:] == (2 * half_side_max + h, 2 * half_side_max + w)

    # random x, y displacement
    t = np.random.randint(half_side_min, half_side_max + 1, size=(2,))
    polarities = np.random.choice([-1, 1], size=(2,), replace=True)
    t *= polarities

    # -x, -y in orig img frame is now -x+half_side_max, -y+half_side_max in new
    t += half_side_max

    img = img[:, t[1] : (t[1] + h), t[0] : (t[0] + w)]
    assert img.shape[1:] == (h, w)

    return img


if __name__ == "__main__":
    data = np.arange(24).reshape((4, 6))
    req = (2, 2)
    start = get_center_start_subscript(data.shape, req)
    print(reshape_by_pad_crop(data, req, start))
    start = get_random_start_subscript(data.shape, req)
    print(reshape_by_pad_crop(data, req, start))
    start = get_random_start_subscript(data.shape, req)
    print(reshape_by_pad_crop(data, req, start))
    start = get_random_start_subscript(data.shape, req)
    print(reshape_by_pad_crop(data, req, start))
    start = get_random_start_subscript(data.shape, req)
    print(reshape_by_pad_crop(data, req, start))
    start = get_random_start_subscript(data.shape, req)
    print(reshape_by_pad_crop(data, req, start))
