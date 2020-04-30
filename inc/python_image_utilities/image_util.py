from itertools import chain, cycle, islice
from math import ceil, floor, isinf, isnan, log10
from pathlib import PurePath
from random import shuffle
from typing import List, Tuple

import cv2
import noise
import numpy as np
from PIL import Image
import scipy.stats

# remove . when testing
from .inc.file_utils.file_utils import get_contents


def adjust_gamma(image, gamma=1.0):
    """Adjusts image gamma. Works on both float and uint8 images. Float images
    should be in the range (0.0, 1.0) for expected behavior.
    """
    igamma = 1.0 / gamma
    if np.issubdtype(image.dtype, np.floating):
        out = image ** igamma
    elif image.dtype == np.uint8:
        lut = np.array([((i / 255.0) ** igamma) * 255.0 for i in range(0, 256)])
        out = cv2.LUT(image, lut.astype(np.uint8))
    else:
        assert False
    return out


def clahe(image, tile_size=(2, 2)):
    """Applies CLAHE equalization to input image. Works on both float and uint8
    images. Works on color images by converting to Lab space and treating the L
    channel as grayscale.
    """
    is_float = np.issubdtype(image.dtype, np.floating)
    if is_float:
        image = float_to_uint8(image)

    is_rgb = image.shape[-1] == 3
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=tile_size)
    if is_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        image[..., 0] = clahe.apply(image[..., 0])
        image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
    else:
        image = clahe.apply(image)

    if is_float:
        image = uint8_to_float(image)
    return image


def consensus(image_stack, threshold="majority"):
    """Builds a consensus label image from a stack of label images. If input is
    NHW1, output is 1HW1, with the same dtype as input.

    Inputs:

    image_stack: A stack of images with shape NHW1 with dtype bool or integer.

    threshold: If a string, must be "majority". This uses a mode for each pixel,
    and can be computationally intensive. This threshold must be chosen if there
    are more than two unique values in the input image_stack. Ties are assigned
    the smallest value in that pixel. If the value is an integer, then it is
    used as the cutoff count for the larger unique value. If the value is a
    float, then it is treated as a proportion of values with the larger unique
    value.
    """
    assert image_stack.dtype == np.bool or np.issubdtype(image_stack.dtype, np.integer)
    image_stack = image_stack.copy().astype(np.uint8)
    unique = np.unique(image_stack)
    if unique.size > 2:
        assert threshold == "majority"

    if isinstance(threshold, float):
        assert 0.0 <= threshold <= 1.0
    elif isinstance(threshold, int):
        assert 0 <= threshold

    out = None
    if threshold == "majority":
        out = scipy.stats.mode(image_stack, axis=0, nan_policy="omit")[0]
    else:
        if isinstance(threshold, float):
            threshold = round(threshold * image_stack.shape[0])
        out = image_stack.sum(axis=0) >= threshold
        out = out.astype(image_stack.dtype)
        out = unique[out.flatten()].reshape(out.shape)
    assert out is not None
    return out.astype(image_stack.dtype)


def float_to_uint8(float_image, float_range=(0.0, 1.0), clip=False):
    """Converts a float image to a uint8 image. Image is rescaled with input
    range equal to float_range. If clip is set to True, values are clipped
    following conversion.
    """
    uint8_image = rescale(float_image, out_range=(0.0, 255.0), in_range=float_range)
    return uint8_image.astype(np.uint8)


def generate_circular_fov_mask(shape, fov_radius, offset=(0, 0)):
    """Generates a circular field of view mask, with the interior of the circle
    included. The circle is assumed to be centered on the image shape.
    """
    center = get_center(shape)
    X, Y = np.meshgrid(np.arange(shape[0]) - center[0], np.arange(shape[1]) - center[1])
    R2 = (X - offset[0]) ** 2 + (Y - offset[1]) ** 2
    fov2 = fov_radius ** 2
    mask = R2 <= fov2
    return mask[..., np.newaxis]


def generate_noise(shape, offsets=None):
    """Generates a grayscale image with single-octave Perlin noise. Useful for
    testing.
    """
    if offsets is None:
        scale = 0.1 * np.array(shape).max()
        offsets = np.random.uniform(-1000 * scale, 1000 * scale, 2)

    octaves = 1
    X, Y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    X = X + offsets[0]
    Y = Y + offsets[1]
    noise_maker = np.vectorize(
        lambda x, y: noise.pnoise2(x / scale, y / scale, octaves=octaves)
    )
    n = noise_maker(X, Y)
    n = ((n - n.min()) / (n.max() - n.min())) * 255
    return n.astype(np.uint8)


def get_center(shape):
    """Returns the coordinates of the center point of an image in pixels,
    rounded down.
    """
    return [floor(x / 2) for x in shape]


def gray2rgb(gray_image):
    """Converts a grayscale image to an RGB image.
    """
    if gray_image.shape[-1] == 1:
        gray_image = gray_image.squeeze(-1)
    return np.stack([gray_image for _ in range(3)], axis=-1)


def lab2rgb(lab_image):
    """Converts an Lab image to an RGB image.
    """
    return cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)


def load(path, force_rgb=False):
    """Loads an image from the supplied path in grayscale or RGB depending on
    the source.
    """
    if PurePath(path).suffix.casefold() in (".tif", ".tiff"):
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = Image.open(path)
        image = np.array(image)

    if image.ndim == 2:
        image = image[..., np.newaxis]

    # convert redundant rgb to grayscale
    if image.shape[2] == 3:
        g_redundant = (image[..., 0] == image[..., 1]).all()
        b_redundant = (image[..., 0] == image[..., 2]).all()
        if (g_redundant and b_redundant) and not force_rgb:
            image = image[..., 0]
            image = image[..., np.newaxis]

    if image.shape[2] == 1 and force_rgb:
        image = np.repeat(image, 3, axis=2)

    assert image.ndim == 3
    assert image.shape[2] in (1, 3)
    return image


def load_images(folder, ext=None) -> Tuple[List[np.array], List[PurePath]]:
    """Loads a folder of images. If an extension is supplied, only images with
    that extension will be loaded. Also returns the filenames of every loaded
    image.
    """
    image_files = get_contents(folder, ext)
    images = []
    names = []
    for image_file in image_files:
        try:
            image = load(str(image_file))
        except Exception as e:
            continue
        images.append(image)
        names.append(image_file)

    return images, names


def mask_images(images, masks):
    """Masks out pixels in an image stack based on the masks. There must be
    either one mask, or the same number of images and masks.
    """
    if masks.shape[0] == 1:
        masks = np.repeat(masks, images.shape[0], axis=0)
    assert masks.shape[0] == images.shape[0]

    masked = images.copy()
    threshold = (masks.max() - masks.min()) / 2.0
    masked[masks <= threshold] = 0
    return masked


def montage(
    images,
    shape=None,
    mode="sequential",
    repeat=False,
    start=0,
    maximum_images=36,
    fill_value=0,
):
    """Generates a montage image from an image stack.

    shape determines the number of images to tile in each dimension. Must be an
    iterable of length 2 containing positive integers, or a single positive
    float, or None. If a float is provided, that value is assumed to be a
    width-to-height aspect ratio and the shape is computed to best fit that
    aspect ratio. If None is provided, the aspect ratio is set to 1. In both
    latter cases, the montage has tiles at least equal to the number of images
    in the stack if maximum_images is not also set. Default is None.

    mode determines how to sample images from the stack. Must be either
    "sequential" or "random". If "sequential", the images are sampled in the
    order they appear in the stack. If "random", then the stack order is
    shuffled before sampling. Default is "sequential".

    repeat determines whether to sample the stack from the start if more images
    are requested than exist in the stack. If repeat is set to False and more
    images are requested than exist, the remaining tiles are filled with zeros,
    i.e. black. Default is False.

    start determines the starting index for sampling the stack. Default is 0.

    maximum_images determines the limit of images to be sampled from the stack,
    regardless of shape. Default is 36.
    """
    image_count = images.shape[0] - start
    if maximum_images is not None:
        image_count = min(maximum_images, image_count)

    if shape is None:
        shape = _optimize_shape(image_count)
    elif isinstance(shape, (int, float)):
        shape = _optimize_shape(image_count, width_height_aspect_ratio=shape)

    indices = list(range(images.shape[0]))

    if mode == "random":
        shuffle(images)
    elif mode == "sequential":
        pass
    else:
        assert False

    if repeat:
        iterator = cycle(indices)
    else:
        iterator = chain(indices, cycle([float("inf")]))

    stop = int(np.array(shape).prod() + start)
    iterator = islice(iterator, start, stop)

    montage = np.stack([_get_image_or_blank(images, i, fill_value) for i in iterator])
    montage = montage.reshape((*shape, *images.shape[1:]))

    a, b = _deinterleave(list(range(0, montage.ndim - 1)))
    a, b = list(a), list(b)
    dim_order = (*a, *b, montage.ndim - 1)
    montage = montage.transpose(dim_order)

    image_shape = np.array(shape) * np.array(images.shape[1:-1])
    image_shape = np.append(image_shape, images.shape[-1])
    return montage.reshape(image_shape)


# TODO fix issue with different channel counts, i.e. gray to rgb for both inputs
def overlay(background, foreground, color, alpha=0.5, beta=0.5, gamma=0.0, clip=False):
    """Applies a color to the supplied grayscale foreground and then blends it
    with the background using mixing ratio parameters alpha and beta. Background
    may be RGB or grayscale. Foreground and background may both be either float
    or uint8. Output is RGB with the same dtype as background. Gamma is a
    constant ratio parameter to add to all pixels.

    Note that if the sum of alpha, beta and gamma is greater than 1.0, clipping
    can occur.
    """
    assert np.issubdtype(background.dtype, np.floating) or background.dtype == np.uint8
    assert background.shape[-1] in (1, 3)
    assert np.issubdtype(foreground.dtype, np.floating) or foreground.dtype == np.uint8
    assert foreground.shape[-1] == 1
    assert len(color) == 3

    if background.dtype == np.uint8:
        background = uint8_to_float(background)
    if background.shape[-1] == 1:
        background = gray2rgb(background)

    if foreground.dtype == np.uint8:
        foreground = uint8_to_float(foreground)
    foreground = gray2rgb(foreground)

    color = np.array(color)
    color = color[np.newaxis][np.newaxis]
    foreground = foreground * color
    out = cv2.addWeighted(foreground, alpha, background, beta, gamma)
    if background.dtype == np.uint8:
        out = float_to_uint8(out, clip=clip)
    return out


def patchify(image_stack, patch_shape, offset=(0, 0), *args, **kwargs):
    """Transforms an image stack into a new stack made of tiled patches from
    images of the original stack. The size of the patches is determined by
    patch_shape. If patch_shape does not evenly divide the image shape, the
    excess is padded with zeros, i.e. black.

    If there are N images of size X by Y, and patches of size M by N are
    requested, then the resulting stack will have N * ceil(X/M) * ceil(Y/N)
    images. The first ceil(X/M) * ceil(Y/N) patches all come from the first
    image, sampled along X first, then Y.

    Returns a tuple consisting of an image stack of patches, the number of
    patches in each dimension, and the padding used. The latter two values are
    used for unpatching.

    Inputs:

    image_stack: Stack of images of shape NHW or NHWC. Assumed to have 2D
    images.

    patch_shape: Spatial shape of patches. All channel information is retained.
    If patch shape is size M by N, then the resulting stack of patches will have
    N * ceil(X/M) * ceil(Y/N) images. Every ceil(X/M) * ceil(Y/N) patches in the
    stack belong to a single image. Patches are sampled along X first, then Y.
    If M divides X and N divides Y, then a non-zero offset will change the
    number of patches.

    offset: Offset is the spatial location of the lower-right corner of the
    top-left patch relative to the image origin. Because the patch boundaries
    are periodic, any value greater than patch_shape in any dimension is reduced
    module patch_shape. The value of any pixels outside the image are assigned
    according to arguments for np.pad().
    """
    assert image_stack.ndim in (3, 4)
    if image_stack.ndim == 3:
        image_stack = image_stack[np.newaxis, ...]

    assert len(patch_shape) == 2
    assert len(offset) == 2

    offset = [o % p for o, p in zip(offset, patch_shape)]

    # determine pre padding
    pre_padding = [((p - o) % p) for o, p in zip(offset, patch_shape)]
    pre_padding = np.append(pre_padding, 0)
    pre_padding = np.insert(pre_padding, 0, 0)
    # compute post padding from whatever is left
    pre_image_shape = [
        s + pre for s, pre in zip(image_stack.shape[1:-1], pre_padding[1:-1])
    ]
    post_padding = patch_shape - np.remainder(pre_image_shape, patch_shape)
    post_padding = np.append(post_padding, 0)
    post_padding = np.insert(post_padding, 0, 0)
    padding = list(zip(pre_padding, post_padding))
    padded = np.pad(image_stack, padding, *args, **kwargs)
    out_padding = padding[1:-1]

    patch_shape = np.array(patch_shape)
    patch_counts = np.array([x // y for x, y in zip(padded.shape[1:-1], patch_shape)])
    patches_shape = _interleave(patch_counts, patch_shape)
    patches_shape = np.append(patches_shape, image_stack.shape[-1])
    patches_shape = np.insert(patches_shape, 0, -1)
    patches = padded.reshape(patches_shape)

    dim_order = _deinterleave(range(1, patches.ndim - 1))
    dim_order = np.append(dim_order, patches.ndim - 1)
    dim_order = np.insert(dim_order, 0, 0)
    patches = patches.transpose(dim_order)

    stacked_shape = (-1, *patch_shape, image_stack.shape[-1])
    patches = patches.reshape(stacked_shape)

    return patches, patch_counts, out_padding


def rescale(
    image, out_range=(0.0, 1.0), in_range=(float("-inf"), float("+inf")), clip=False
):
    """Rescales image from in_range to out_range while retaining input dtype. If
    clip is set to True, the resulting image values are clipped to out_range.
    """
    lo = in_range[0]
    if lo == float("-inf") or isnan(lo):
        lo = np.min(image)

    hi = in_range[1]
    if hi == float("+inf") or isnan(hi):
        hi = np.max(image)

    assert not isinf(lo) and not isnan(lo)
    assert not isinf(hi) and not isnan(hi)
    assert lo <= hi

    if lo == hi:
        return image

    med = (image - lo) / (hi - lo)
    out = med * (out_range[1] - out_range[0]) + out_range[0]
    if np.issubdtype(image.dtype, np.integer):
        out = np.round(out)
    if clip:
        out = np.clip(out, out_range[0], out_range[1])
    return out.astype(image.dtype)


def resize(image, method="area", size=None, scale=1.0):
    METHODS = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "area": cv2.INTER_AREA,
        "cubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
    }
    assert method in METHODS

    if size is None:
        if isinstance(scale, float):
            scale = (scale, scale)
        assert isinstance(scale, tuple)
        assert len(scale) == 2
        assert isinstance(scale[0], float)
        assert isinstance(scale[1], float)

        out = cv2.resize(image, (0, 0), None, scale[0], scale[1], METHODS[method])
    elif scale is None:
        assert isinstance(size, tuple)
        assert len(size) == 2
        assert isinstance(size[0], int)
        assert isinstance(size[1], int)

        out = cv2.resize(image, size)
    elif scale is None and size is None:
        assert False
    else:
        assert False
    return out


def rgb2gray(rgb_image):
    """Converts an RGB image to grayscale.
    """
    return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)


def rgb2lab(rgb_image):
    """Converts an RGB image to Lab.
    """
    return cv2.cvtColor(rgb_image, cv2.COLOR_LAB2RGB)


def save(path, image):
    """Saves an image to disk at the location specified by path.
    """
    if np.issubdtype(image.dtype, np.floating):
        image = float_to_uint8(image)
    if image.shape[-1] == 1:
        image = image.squeeze(axis=-1)
    im = Image.fromarray(image)
    im.save(str(path))


def save_images(paths, image_stack):
    """Saves an image stack to disk as individual images using save() with index
    appended to the supplied file name, joined by delimiter.

    paths is the file paths for images to be written.

    images is a Numpy array whose shape is of the form (NHWC) where N is the
    number of images, HW are spatial dimensions, and C are the channels. N may
    be any positive number, H and W may be any positive numbers, and C must be 1
    or 3.
    """
    assert image_stack.ndim == 4
    assert len(paths) == len(image_stack)
    for path, image in zip(paths, image_stack):
        save(path, image)


def stack(images):
    """Converts a single image or an iterable of images to an image stack. An
    image stack is a numpy array whose first dimension is the image index.
    """
    if type(images) is np.ndarray:
        images = (images,)
    images = [image[..., np.newaxis] if image.ndim == 2 else image for image in images]
    return np.stack(images)


def standardize(images):
    """Standardizes an (N+1)-D block of N-D images by the usual method, i.e.
    (x-u)/s.

    This function is intended to be used just before application of a machine
    learning model, or for training such a model. There is no guarantee the
    output will be in the usual (0.0, 1.0) range.
    """
    s = np.std(images)
    u = np.mean(images)
    standardized = (images - u) / s
    return standardized


def uint8_to_float(uint8_image, float_range=(0.0, 1.0), clip=False):
    """Converts a uint8 image to a float image. If clip is set to True, values
    are clipped to float_range."""
    return rescale(
        uint8_image.astype(np.float), in_range=(0.0, 255.0), out_range=float_range
    )


def unpatchify(patches, patch_counts, padding):
    """Inverse of patchify(). Transforms an image stack of patches produced
    using patchify() back into an image stack of the same shape as the original
    images. Requires the patch_count and padding returned by patchify().
    """
    chunk_len = np.array(patch_counts).prod()
    base_shape = np.array(patch_counts) * np.array(patches.shape[1:-1])
    image_shape = np.append(base_shape, patches.shape[-1])
    image_count = patches.shape[0] // chunk_len
    chunk_shape = (*patch_counts, *patches.shape[1:])
    images = []
    for i in range(image_count):
        chunk = patches[i * chunk_len : (i + 1) * chunk_len]
        chunk = np.reshape(chunk, chunk_shape)
        chunk = np.transpose(chunk, (0, 2, 1, 3, 4))
        images.append(np.reshape(chunk, image_shape))
    images = np.stack(images)
    padding = list(zip(*padding))
    pre_padding = padding[0]
    post_padding = padding[1]
    space_shape = [
        base - pre - post
        for base, pre, post in zip(base_shape, pre_padding, post_padding)
    ]
    # space_shape = base_shape - padding
    slices = [slice(pre, pre + x) for pre, x in zip(pre_padding, space_shape)]
    slices.append(slice(None))
    slices.insert(0, slice(None))
    images = images[tuple(slices)]
    return images


def show(image, tag="UNLABELED_WINDOW"):
    """Displays an image in a new window labeled with tag.
    """
    assert image.ndim in (2, 3)
    if image.ndim == 3:
        assert image.shape[-1] in (1, 3)

    cv2.namedWindow(tag, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(tag, image.shape[0:2][::-1])
    if image.shape[-1] == 3:
        image = image[..., ::-1]
    cv2.imshow(tag, image)
    cv2.waitKey(1)


def _deinterleave(c):
    """Separates two interleaved sequences into a tuple of two sequences of the same type.
    """
    a = c[0::2]
    b = c[1::2]
    return a, b


def _get_image_or_blank(images, index, fill_value=0):
    try:
        return images[index]
    except Exception as e:
        return np.zeros(images.shape[1:], dtype=images.dtype) + fill_value


def _interleave(a, b):
    """Interleaves two sequences of the same type into a single sequence.
    """
    c = np.empty((a.size + b.size), dtype=a.dtype)
    c[0::2] = a
    c[1::2] = b
    return c


def _optimize_shape(count, width_height_aspect_ratio=1.0):
    """Computes the optimal X by Y shape of count objects given a desired
    width-to-height aspect ratio.
    """
    N = count
    W = np.arange(1, N).astype(np.uint32)
    H = np.ceil(N / W).astype(np.uint32)
    closest = np.argmin(np.abs((W / H) - width_height_aspect_ratio))
    return H[closest], W[closest]
