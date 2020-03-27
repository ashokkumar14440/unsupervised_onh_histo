import unittest
from math import floor

import numpy as np

from patch_extract import *
from image_util import *


class Test(unittest.TestCase):
    def setUp(self):
        self.side_len = 200
        self.base_shape = (self.side_len, self.side_len)

        self.rgb = np.moveaxis(np.indices(self.base_shape), 0, -1)
        self.rgb = np.concatenate((self.rgb, np.zeros(self.base_shape + (1,))), axis=2)
        self.rgb = self.rgb.astype(np.uint8)
        self.rgb_shape = self.rgb.shape

        self.fov_radius_ratio = 0.45
        self.fov_offset = (-35, 45)
        self.fov_radius = floor(self.side_len * self.fov_radius_ratio)

        self.mask = generate_circular_fov_mask(
            self.base_shape, self.fov_radius, self.fov_offset
        )
        self.mask_shape = self.mask.shape

        self.patch_ratio = 0.2
        self.patch_shape = tuple(floor(x * self.patch_ratio) for x in self.base_shape)

        self.random_count = int(1e5)  # leave alone for random checks

    def tearDown(self):
        pass

    def create_cover(self):
        return np.zeros(self.base_shape).astype(np.bool)

    def report_coverage(self, coverage):
        im_cover = coverage.sum() / self.mask.size
        print("Image coverage: {:.2%}".format(im_cover))
        return im_cover

    def report_fov_coverage(self, coverage):
        fov_cover = (coverage & self.mask).sum() / self.mask.sum()
        print("FOV coverage: {:.2%}".format(fov_cover))
        return fov_cover

    def test_generate_random_patch(self):
        cover = self.create_cover()
        for i in range(self.random_count):
            patch, mask_patch = generate_random_patch(
                self.rgb, self.patch_shape, auxiliary_image=self.mask
            )
            cover[patch[..., 0], patch[..., 1]] = True
        coverage = self.report_coverage(cover)
        self.assertGreaterEqual(coverage, 0.98)

    def test_generate_random_fov_patch(self):
        cover = self.create_cover()
        for i in range(self.random_count):
            patch, mask_patch = generate_random_patch(
                self.rgb, self.patch_shape, self.mask, auxiliary_image=self.mask
            )
            cover[patch[..., 0], patch[..., 1]] = True
        coverage = self.report_fov_coverage(cover[..., np.newaxis])
        self.assertGreaterEqual(coverage, 0.98)


if __name__ == "__main__":
    unittest.main()
