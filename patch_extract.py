import random

from image_util import get_center


def extract_patch(image, patch_shape, patch_center):
    """Extracts a rectangular patch from an image.
    """
    x, y = get_corners(patch_center, patch_shape)
    return image[x[0] : x[1], y[0] : y[1], ...]


def generate_random_image_point(image_shape, patch_shape):
    """Randomly picks a point uniformly from the interior of an image.
    """
    return [
        random.randint(p, i - p - 1)
        for p, i in zip(get_center(patch_shape), image_shape)
    ]


def generate_random_patch(
    image, patch_shape, fov_mask=None, auxiliary_image=None, attempt_limit=100
):
    """Generates a random patch contained entirely within the image bounds. The
    patch is extracted from the input image and returned. If a field of
    view mask is supplied, the patch must also be entirely within the mask. If
    no suitable random patches can be found after attempt_limit tries, None is
    returned. Otherwise the extracted patch is returned. If an auxiliary_image
    is provided, a patch from the same region is returned.
    """
    image_shape = image.shape[0:2]
    attempts = 0
    while True:
        if attempts >= attempt_limit:
            return None, None
        point = generate_random_image_point(image_shape, patch_shape)
        if fov_mask is not None and not fov_mask[point[0], point[1]]:
            attempts += 1
            continue
        patch = extract_patch(image, patch_shape, point)
        if auxiliary_image is not None:
            aux_patch = extract_patch(auxiliary_image, patch_shape, point)
        else:
            aux_patch = None
        return patch, aux_patch


def get_corners(center, shape, origin=[0, 0], fn=lambda x: x):
    """Returns coordinates of four corners of axially-aligned rectangle from the
    origin.
    """
    mid = get_center(shape)
    pos = [fn(c + m - o) for c, m, o in zip(center, mid, origin)]
    neg = [fn(c - m - o) for c, m, o in zip(center, mid, origin)]
    x = [neg[0], pos[0]]
    y = [neg[1], pos[1]]
    return x, y


def get_corner_distances(center, shape, origin=[0, 0]):
    """Returns squared distance of four corners of an axially-aligned rectangle
    from the origin. Return value is sorted ascending by y first, then x.
    """
    x, y = get_corners(center, shape, origin, lambda x: x ** 2)
    return [x[i] + y[j] for i in [0, 1] for j in [0, 1]]
