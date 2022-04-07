import functools
import inspect
import itertools
from math import floor, isinf, isnan
from pathlib import Path, PurePath
from random import shuffle
from typing import Callable, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import scipy.stats
import skimage
import tifffile.tifffile as tf
from PIL import Image

from inc.file_utils.file_utils import get_contents

"""
This library is intended for 2D images, and stacks of 2D images. It is not
suitable for general ND arrays.

When using this library, all images must have three dimensions (h,w,c). All
image stacks must have four dimensions (n,h,w,c). Most functions will handles
this automatically, but not all.
"""


PathLike = Union[str, Path, PurePath]
Number = Union[int, float]


def _as_colorspace(space: str, fromspace: str = "rgb") -> Callable:
    """
    Wraps a function in a roundtrip colorspace conversion.
    """

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(image, *args, **kwargs):
            kwargs.update(_get_default_args(fn))
            assert "use_signed_negative" in kwargs
            use_signed_negative = kwargs["use_signed_negative"]
            assert _is_color(image)
            image = to_colorspace(
                image,
                fromspace=fromspace,
                tospace=space,
                use_signed_negative=use_signed_negative,
            )
            image = fn(image, *args, **kwargs)
            image = to_colorspace(
                image,
                fromspace=space,
                tospace=fromspace,
                use_signed_negative=use_signed_negative,
            )
            assert _is_color(image)
            return image

        return wrapper

    return decorator


def _as_dtype(dtype) -> Callable:
    """
    Wraps a function in round-trip dtype conversion. Useful for decorating
    functions that require a specific dtype.
    """

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(image, *args, **kwargs):
            kwargs = _update_with_defaults(fn, kwargs)
            assert "use_signed_negative" in kwargs
            use_signed_negative = kwargs["use_signed_negative"]
            old_dtype = image.dtype
            out = to_dtype(image, dtype=dtype, negative_in=use_signed_negative)
            out = fn(out, *args, **kwargs)
            # print(type(out))
            out = to_dtype(out, dtype=old_dtype, negative_out=use_signed_negative)
            assert out.dtype == old_dtype
            return out

        return wrapper

    return decorator


def _copy(fn) -> Callable:
    """
    Wraps a function with a copy command.
    """

    @functools.wraps(fn)
    def wrapper(image: np.ndarray, *args, **kwargs) -> np.ndarray:
        kwargs = _update_with_defaults(fn, kwargs)
        # print(kwargs)
        out = image.copy()
        out = fn(out, *args, **kwargs)
        return out

    return wrapper


@_as_dtype(np.float64)
@_copy
def adjust_gamma(
    image: np.ndarray, gamma: float = 1.0, use_signed_negative: bool = False
) -> np.ndarray:
    """
    Adjusts image gamma. Channels are adjusted uniformly.

    Inputs:
    1) image - HWC input image to adjust.
    2) gamma - Float denoting gamma value of output relative to input gamma of
        1.0
    3) use_signed_negative - Consider negative values in signed input types.

    Output:
    1) Grayscale or RGB ND input image with adjusted gamma.
    """
    assert _is_image(image)
    assert _is_gray(image) or _is_color(image)

    image = image ** (1.0 / gamma)

    assert _is_image(image)
    assert _is_gray(image) or _is_color(image)

    return image


@_copy
def clahe(
    image,
    clip_limit: float = 0.01,
    tile_size: Tuple[int, int] = (8, 8),
    use_signed_negative: bool = False,
):
    """
    Applies CLAHE equalization to input image. Works on color images by
    converting to Lab space, performing clahe on the L channel, and converting
    back to RGB. Note that the underlying image type used in the clahe process
    is uint16 which may create a narrowing conversion for some input dtypes.
    Returns the same dtype as input.

    Inputs:
    1) image - HWC input image to adjust.
    2) tile_size - 2-tuple integer size of tiles to perform clahe on.
    3) use_signed_negative - Consider negative values in signed input types.

    Output:
    1) HWC image of same dtype as input image.
    """
    assert _is_image(image)
    if _is_color(image):
        out = _clahe_rgb(
            image,
            clip_limit=clip_limit,
            tile_size=tile_size,
            use_signed_negative=use_signed_negative,
        )
    else:
        out = _clahe_channel(
            image,
            clip_limit=clip_limit,
            tile_size=tile_size,
            use_signed_negative=use_signed_negative,
        )
    assert _is_image(image)
    assert out.shape == image.shape

    return out


@_copy
def consensus(
    stack: np.ndarray, threshold: Optional[Union[str, float, int]] = None
) -> np.ndarray:
    """
    Builds a consensus label image from a stack of label images. If input is
    NHW1 output is HW1, with the same dtype as input. Input channels must be
    one.

    Inputs:
    1) stack - NHW1 stack of images with dtype bool or integer.
    2) threshold - String, float ratio or integer value
        1) stack has integer dtype
            1) "majority" - (integer dtype required, default) Finds the mode for each pixel.
                Computationally intensive. It is recommended to only use dtype np.uint8 as input.
        2) stack has bool dtype
            1) float - (bool dtype required) Ratio of image stack N in range
               [0.0, 1.0]. Finds pixels where more than the fraction supplied are
               True.
            2) integer - (bool dtype required) Absolute count of images in stack
               which are True is more than the supplied value.

    Output:
    1) HW1 image with same dtype as input
    """
    # print(threshold)
    # print(type(threshold))
    assert _is_stack(stack)
    assert _is_gray(stack)

    dtype = stack.dtype
    out = stack.copy()
    assert np.issubdtype(dtype, np.integer) or dtype == np.bool
    if threshold is None:
        if np.issubdtype(dtype, np.integer):
            threshold = "majority"
        elif dtype == np.bool:
            threshold = 0.5
        else:
            assert False
    assert isinstance(threshold, (str, float, int))
    if isinstance(threshold, float):
        assert 0.0 <= threshold <= 1.0
        threshold = round(threshold * stack.shape[0])
    if isinstance(threshold, str):
        assert threshold.casefold() == "majority"
        assert np.issubdtype(dtype, np.integer)
    elif isinstance(threshold, int):
        assert 0 <= threshold
    else:
        assert False

    if threshold == "majority":
        out = scipy.stats.mode(out, axis=0, nan_policy="omit")[0]
        out = out[0, ...]
    else:
        out = out.sum(axis=0) > threshold
        out = out.astype(dtype)

    assert out.dtype == dtype
    assert _is_image(out)
    assert _is_gray(out)

    return out


def load(path: PathLike, force_rgb=False):
    """
    Loads an image from the supplied path in grayscale or RGB depending on the
    source. If the source is RGB and has redundant channels, the image will be
    converted to grayscale. If force_rgb is True, the image will be returned
    with 3 channels. If the image has only one channel, then all channels will
    be identical.
    """
    if PurePath(path).suffix.casefold() in (".tif", ".tiff"):
        image = tf.imread(path)
    else:
        image = Image.open(str(path))
        image = np.array(image)

    image = _add_channel_dim(image)

    # convert redundant rgb to grayscale
    if _is_color(image):
        g_redundant = (image[..., 0] == image[..., 1]).all()
        b_redundant = (image[..., 0] == image[..., 2]).all()
        if (g_redundant and b_redundant) and not force_rgb:
            image = image[..., 0]
            image = image[..., np.newaxis]

    if _is_gray(image) and force_rgb:
        image = np.repeat(image, 3, axis=2)

    assert image.ndim == 3
    assert _is_gray(image) or _is_color(image)
    return image


def montage(
    images,
    image_counts=None,
    mode="sequential",
    repeat=False,
    start=0,
    maximum_images=None,
    fill_value=0,
):
    """
    Generates a montage image from an image stack.

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
    # TODO add stitch border width, color

    assert images.ndim == 4

    image_count_to_guide_shape = images.shape[0] - start
    if maximum_images is not None:
        # TODO may be redundant if shape is provided
        image_count_to_guide_shape = min(maximum_images, image_count_to_guide_shape)

    if image_counts is None:
        image_counts = _optimize_shape(image_count_to_guide_shape)
    elif isinstance(image_counts, (int, float)):
        image_counts = _optimize_shape(
            image_count_to_guide_shape, width_height_aspect_ratio=image_counts
        )

    indices = list(range(images.shape[0]))

    if mode == "random":
        shuffle(indices)
    elif mode == "sequential":
        pass
    else:
        assert False

    image_to_sample_count = int(np.array(image_counts).prod())
    stop = min(image_to_sample_count, image_count_to_guide_shape) + start
    indices = itertools.islice(indices, start, stop)

    if repeat:
        indices = itertools.cycle(indices)
    else:
        indices = itertools.chain(indices, itertools.cycle([float("inf")]))

    indices = itertools.islice(indices, 0, image_to_sample_count)

    montage = np.stack([_get_image_or_blank(images, i, fill_value) for i in indices])
    montage_shape = [c * s for c, s in zip(image_counts, images.shape[1:-1])]
    montage_shape.append(images.shape[-1])
    montage_shape = tuple(montage_shape)
    montage = unpatchify_image(montage, image_shape=montage_shape, offset=(0, 0))

    return montage


def overlay(
    background,
    foreground,
    color,
    alpha=0.5,
    beta=0.5,
    gamma=0.0,
    use_signed_negative: bool = False,
):
    """
    Applies a color to the supplied grayscale foreground and then blends it with
    the background using mixing ratio parameters alpha and beta. Background may
    be RGB or grayscale. Foreground and background may both be either float or
    uint*. Output is RGB with the same dtype as background. Gamma is a constant
    ratio parameter to add to all pixels.

    Note that if the sum of alpha, beta and gamma is greater than 1.0, clipping
    can occur.

    Note that underlying computation is performed on int32, which may create a
    narrowing conversion. Returned dtype is same as background.
    """
    assert _is_image(background)
    assert _is_gray(background) or _is_color(background)
    assert _is_image(foreground)
    assert _is_gray(foreground)
    assert len(color) == 3
    for c in color:
        assert 0.0 <= c <= 1.0

    dtype = background.dtype
    background = background.copy()
    background = to_dtype(background, dtype=np.int32, negative_in=use_signed_negative)
    if _is_gray(background):
        background = gray_to_color(background, color=[1.0, 1.0, 1.0])

    foreground = foreground.copy()
    foreground = to_dtype(foreground, dtype=np.int32, negative_in=use_signed_negative)
    foreground = gray_to_color(foreground, color=color)

    out = cv2.addWeighted(
        src1=foreground, src2=background, alpha=alpha, beta=beta, gamma=gamma
    )
    out = to_dtype(out, dtype=dtype, negative_out=use_signed_negative)

    return out


@_as_dtype(np.float64)
@_copy
def gray_to_color(
    image: np.ndarray, color: Sequence[float], use_signed_negative: bool = False
) -> np.ndarray:
    assert _is_image(image)
    assert _is_gray(image)

    assert len(color) == 3
    for c in color:
        assert 0.0 <= c <= 1.0

    out = image.squeeze()
    out = skimage.color.gray2rgb(out)
    out = out * np.array(color)

    assert _is_image(out)
    assert _is_color(out)

    return out


def patchify_stack(
    stack: np.ndarray,
    patch_shape: Tuple[int, int],
    offset: Tuple[int, int],
    *args,
    **kwargs
) -> np.ndarray:
    """
    Transforms an image stack into a stack of patch stacks. Each patch has shape
    of patch_shape.

    Inputs:
        1. stack - (n,h,w,c) stack of images to transform into patches.
        2. patch_shape - (h,w) shape of patches. Note channel dimension is left
           unchanged.
        3. offset - (h,w) position of offset of top-left corner of a patch. Pre-
           and post-padding will occur to ensure complete coverage of the input.
        4. *args, **kwargs - forwarded to np.pad()

    Outputs:
        1. (k,m,h,w,c) stack of k patch stacks with m patches.
    """
    assert _is_stack(stack)
    return np.stack(
        [
            patchify_image(im, patch_shape=patch_shape, offset=offset, *args, **kwargs)
            for im in stack
        ]
    )


def patchify_image(
    image: np.ndarray,
    patch_shape: Tuple[int, int],
    offset: Tuple[int, int] = (0, 0),
    *args,
    **kwargs
) -> np.ndarray:
    """
    Transforms an image into a patch stack. Each patch has shape of patch_shape.

    Inputs:
        1. image - (h,w,c) stack of images to transform into patches.
        2. patch_shape - (h,w) shape of patches. Note channel dimension is left
           unchanged.
        3. offset - (h,w) position of offset of top-left corner of a patch. Pre-
           and post-padding will occur to ensure complete coverage of the input.
        4. *args, **kwargs - forwarded to np.pad()

    Outputs:
        1. (n,h,w,c) stack of n patches.
    """
    assert _is_image(image)
    assert len(patch_shape) == 2
    assert len(offset) == 2

    # prepare
    offset = tuple([o % s for o, s in zip(offset, patch_shape)])

    # padding
    padding = _compute_patch_padding(
        patch_shape=patch_shape, offset=offset, image_space_shape=image.shape[:-1]
    )
    padded_image = np.pad(image, padding, *args, **kwargs)

    # patching
    stacked_shape = (-1, *patch_shape, image.shape[-1])
    patches = padded_image.reshape(stacked_shape)

    return patches


def rescale(
    image: np.ndarray,
    out_range: Optional[Tuple[Optional[Number], Optional[Number]]] = None,
    in_range: Optional[Tuple[Optional[Number], Optional[Number]]] = None,
    clip: bool = False,
    dtype=None,
    negative_allowed_out: bool = False,
) -> np.ndarray:
    """
    Rescales image values from in_range to out_range. Note that in_range and
    out_range need not have any relation to the underlying dtypes or each other.
    This means, for example, that a uint8 image can have values from 0-127
    scaled out to 0-255, washing out values 128-255. An optionally supplied
    dtype may be used to change the dtype. The function uses np.float64 as a
    common language. All scaling is performed linearly.

    If a dtype is supplied, the output image will have that dtype. Do not use
    this function for thresholding with dtype np.bool. Output images are
    ultimately converted using as_type().

    If clip is set to True, the resulting image values are clipped to the
    narrower of out_range or the limits of the output dtype. Naturally, values
    are always clipped to the limits of the output dtype.

    Default behavior is to scale the image values to the full range of the
    output dtype. Default output dtype is the input dtype.

    Inputs:
    1) image - 2D image whose values will be rescaled.
    2) out_range - (optional) Range of values that in_range is mapped to. Default is range
       of output dtype. See below for more information on allowed values.
    3) in_range - (optional) Range of values to map to out_range. Default is range of
       values in image. See below for more information on allowed values.
    4) clip - clips values to narrower of out_range and dtype range if True. Default is False
    5) dtype - (optional) Output will have this dtype. May cause narrowing
       conversion if care is not taken with in_range/out_range relationship.
    6) negative_allowed_out - If True, instead of zero, uses negative minimum in
       default range for signed integer output dtypes. Default is False.

    Allowed values of *_range inputs:
    1) If *_range is None, the default range is used.
    2) If *_range is a tuple with at least one: None; (-/+)inf; or NaN; the
       default range min or max is used for 1st and 2nd tuple position,
       respectively of (-/+).
    3) First tuple value of range (minimum) must be less than or equal to second
       tuple value (maximum).
    4) If minimum and maximum are equal then the image is returned unchanged.
    5) Ranges of dtypes are full width for integral types and [0.0, 1.0] for
       floating types.
    """
    if dtype is None:
        dtype = image.dtype
    assert dtype is not None

    image = image.copy().astype(np.float64)

    # IN RANGE
    if in_range is None:
        in_range = (None, None)

    in_lo = in_range[0]
    if in_lo is None or in_lo == float("-inf") or isnan(in_lo):
        in_lo = np.nanmin(image)
    assert not isinf(in_lo) and not isnan(in_lo) and in_lo is not None

    in_hi = in_range[1]
    if in_hi is None or in_hi == float("+inf") or isnan(in_hi):
        in_hi = np.nanmax(image)
    assert not isinf(in_hi) and not isnan(in_hi) and in_hi is not None

    assert in_lo <= in_hi
    in_range = (in_lo, in_hi)

    # OUT RANGE
    if out_range is None:
        out_range = (None, None)

    out_lo = out_range[0]
    if out_lo is None or out_lo == float("-inf") or isnan(out_lo):
        out_lo = _get_dtype_range(dtype, allow_negative=negative_allowed_out)[0]
    assert not isinf(in_lo) and not isnan(in_lo) and in_lo is not None

    out_hi = out_range[1]
    if out_hi is None or out_hi == float("+inf") or isnan(out_hi):
        out_hi = _get_dtype_range(dtype, allow_negative=negative_allowed_out)[1]
    assert not isinf(out_hi) and not isnan(out_hi) and out_hi is not None

    assert out_lo <= out_hi
    out_range = (out_lo, out_hi)

    out = image
    if in_range[0] != in_range[1] and out_range[0] != out_range[1]:
        out = (out - in_range[0]) / (in_range[1] - in_range[0])
        out = out * (out_range[1] - out_range[0]) + out_range[0]

    if clip:
        out = np.clip(out, out_range[0], out_range[1])

    return out.astype(dtype)


def resize(image, method="linear", size=None, scale=1.0):
    """
    Resizes image to a specific size or by a scaling factor using supplied method.

    Inputs:
    1) image - 2D image to resize
    2) method - one of:
        1) nearest - resized pixel will be nearest neighbor pixel from original image
        2) linear - (default) bilinear interpolation
        3) cubic - bicubic interpolation
        4) area - resample by pixel area relation, moire free for image shrinking, not suitable for image growing
        5) lanczos4 - 8x8 lanczos interpolation
        6) nearest_exact - bit exact nearest neighbor interpolation, same as PIL, scikit-image, MATLAB
        7) linear_exact - bit exact bilinear interpolation
    """
    METHODS = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "area": cv2.INTER_AREA,
        "lanczos4": cv2.INTER_LANCZOS4,
        "linear_exact": cv2.INTER_LINEAR_EXACT,
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

        out = cv2.resize(image, size, interpolation=METHODS[method])
    elif scale is None and size is None:
        assert False
    else:
        assert False
    return out


def save(image, path: PathLike, dtype=None):
    """
    Saves an image to disk at the location specified by path.
    """
    image = image.copy()
    if dtype is not None:
        image = to_dtype(image, dtype=dtype)
    if image.shape[-1] == 1:
        image = image.squeeze(axis=-1)
    if PurePath(path).suffix.casefold() in (".tif", ".tiff"):
        image = tf.imwrite(path, data=image)
    else:
        im = Image.fromarray(image)
        im.save(str(path))


def show(image, tag="UNLABELED_WINDOW"):
    """
    Displays an image in a new window labeled with tag.
    """
    assert image.ndim in (2, 3)
    if image.ndim == 3:
        assert _is_gray(image) or _is_color(image)

    cv2.namedWindow(tag, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(tag, image.shape[0:2][::-1])
    if _is_color(image):
        image = image[..., ::-1]
    cv2.imshow(tag, image)
    cv2.waitKey(1)


def stack(images):
    """
    Converts a single image or an iterable of images to an image stack. An image
    stack is a numpy array whose first dimension is the image index.
    """
    if type(images) is np.ndarray:
        images = (images,)
    images = [image[..., np.newaxis] if image.ndim == 2 else image for image in images]
    return np.stack(images)


def standardize(images):
    """
    Standardizes an (N+1)-D block of N-D images by the usual method, i.e.
    (x-u)/s.

    This function is intended to be used just before application of a machine
    learning model, or for training such a model. There is no guarantee the
    output will be in the usual (0.0, 1.0) range.
    """
    s = np.std(images)
    u = np.mean(images)
    standardized = (images - u) / s
    return standardized


@_as_dtype(np.float64)
def to_colorspace(
    image: np.ndarray, fromspace: str, tospace: str, use_signed_negative: bool = False
) -> np.ndarray:
    """
    Uses skimage colorspaces in `skimage.color.*`.
    """
    out = skimage.color.convert_colorspace(
        image, fromspace=fromspace.upper(), tospace=tospace.upper()
    )
    out = _add_channel_dim(out)
    return out


def to_dtype(
    image: np.ndarray, dtype, negative_in: bool = True, negative_out: bool = True
) -> np.ndarray:
    """
    Converts image to desired dtype by rescaling input dtype value range into
    output dtype value range. If negative_* is True, the range [dtype_min,
    dtype_max] is used for signed integer types, otherwise [0, dtype_max] is
    used. Allowed dtypes include any subdtype of np.integer or np.floating.

    Inputs:
    1) image - 2D image to convert to dtype.
    2) dtype - Desired output dtype.
    3) negative_in - (bool) Use full signed integer range for input dtype.
       Default is True.
    4) negative_out - (bool) Use full signed integer range for output dtype.
       Default is True.

    IMPORTANT: For single-channel images, most file formats only support
    unsigned integer types. TIFF allow signed integer types. For multi-channel
    images, most file formats only uint8 images.
    """
    in_range = _get_dtype_range(image.dtype, allow_negative=negative_in)
    out_range = _get_dtype_range(dtype, allow_negative=negative_out)
    return rescale(image, out_range=out_range, in_range=in_range, dtype=dtype)


@_as_dtype(np.float64)
@_copy
def to_gray(image: np.ndarray, use_signed_negative: bool = False):
    assert _is_image(image)
    if not _is_color(image):
        assert _is_gray(image)
        return image

    out = skimage.color.rgb2gray(image)
    out = _add_channel_dim(out)

    assert _is_image(out)
    assert _is_gray(out)

    return out


def unpatchify_stack(
    patches: np.ndarray,
    stack_shape: Tuple[int, int, int, int],
    offset: Tuple[int, int],
) -> np.ndarray:
    """
    Inverse of patchify_stack(). Transforms output of patchify_stack() back into
    a stack of images of the same shape input to patchify_stack().

    Inputs:
        1. patches - (m,n,h,w,c) stack of m patch stacks of n patches
        2. stack_shape - (n,h,w,c) of original stack
        3. offset - (h,w) position of offset passed to patchify_stack()

    Outputs:
        1. stack of k images with shape stack_shape
    """
    assert patches.ndim == 5
    return np.stack(
        [
            unpatchify_image(im, image_shape=stack_shape[1:], offset=offset)
            for im in patches
        ]
    )


def unpatchify_image(
    patches: np.ndarray, image_shape: Tuple[int, int, int], offset: Tuple[int, int]
):
    """
    Inverse of patchify(). Transforms an patch stack into an image of shape
    image_shape.

    Inputs:
        1. patches - (n,h,w,c) patch stack of n patches
        2. image_shape (h,w,c) of original image
        3. offset - (h,w) position of offset passed to patchify_image()
    """
    assert _is_stack(patches)
    assert len(image_shape) == 3

    # prepare
    patch_shape = patches.shape[1:-1]

    # rebuild padded image
    padding = _compute_patch_padding(
        patch_shape=patch_shape, offset=offset, image_space_shape=image_shape[:-1]
    )
    padded_image_shape = [x + p[0] + p[1] for p, x in zip(padding, image_shape)]
    padded_image = patches.reshape(padded_image_shape)

    # extract original image
    starts = [(s - o) % s for s, o in zip(patch_shape, offset)]
    slicer = [slice(st, st + x) for st, x in zip(starts, image_shape)]
    slicer.append(slice(None))
    slicer = tuple(slicer)
    image = padded_image[slicer]

    return image


def _add_channel_dim(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = image[..., np.newaxis]
    return image


@_as_colorspace("hsv")
def _clahe_rgb(image, *args, **kwargs):
    assert _is_color(image)
    out = _add_channel_dim(image[..., -1])
    out = _clahe_channel(out, *args, **kwargs)
    image[..., -1] = out[..., -1]
    assert _is_color(image)

    return image


@_as_dtype(np.uint16)
def _clahe_channel(
    image,
    clip_limit: float,
    tile_size: Tuple[int, int],
    use_signed_negative: bool = False,
):
    assert _is_gray(image)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    out = clahe.apply(image)
    out = _add_channel_dim(out)
    assert _is_gray(out)

    return out


def _compute_patch_padding(
    patch_shape: Tuple[int, int],
    offset: Tuple[int, int],
    image_space_shape: Tuple[int, int],
) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    pre_pad = [(s - o) % s for s, o in zip(patch_shape, offset)]
    pre_pad.append(0)  # channels
    post_pad = [
        (s - p - x) % s for s, p, x in zip(patch_shape, pre_pad, image_space_shape)
    ]
    post_pad.append(0)  # channels
    padding = tuple(zip(pre_pad, post_pad))
    return padding


def _get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def _get_dtype_range(dtype, allow_negative=False):
    if np.issubdtype(dtype, np.integer):
        ii = np.iinfo(dtype)
        if not allow_negative and np.issubdtype(dtype, np.signedinteger):
            value = (0, ii.max)
        else:
            value = (ii.min, ii.max)
    elif np.issubdtype(dtype, np.floating):
        value = (0.0, 1.0)
    elif dtype == np.bool:
        value = (False, True)
    else:
        raise TypeError("supplied dtype is unsupported: {:s}".format(str(dtype)))
    return value


def _get_image_or_blank(images, index, fill_value=0):
    try:
        return images[index]
    except Exception as e:
        return np.zeros(images.shape[1:], dtype=images.dtype) + fill_value


def _is_color(image: np.ndarray) -> bool:
    if image.ndim == 3:
        is_rgb = image.shape[-1] == 3
    else:
        is_rgb = False
    return is_rgb


def _is_gray(image: np.ndarray) -> bool:
    if image.ndim > 2:
        is_gray = image.shape[-1] == 1
    else:
        is_gray = False
    return is_gray


def _is_image(image):
    return image.ndim == 3


def _is_stack(image):
    return image.ndim == 4


def _optimize_shape(count, width_height_aspect_ratio=1.0):
    """
    Computes the optimal X by Y shape of count objects given a desired
    width-to-height aspect ratio.
    """
    N = count
    W = np.arange(1, N).astype(np.uint32)
    H = np.ceil(N / W).astype(np.uint32)
    closest = np.argmin(np.abs((W / H) - width_height_aspect_ratio))
    return H[closest], W[closest]


def _update_with_defaults(fn: Callable, kwargs: dict) -> dict:
    for k, v in _get_default_args(fn).items():
        if k not in kwargs:
            kwargs[k] = v
    return kwargs
