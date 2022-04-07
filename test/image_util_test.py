import unittest
from math import ceil
from pathlib import Path, PurePath

import numpy as np
from image_util import *


class Test(unittest.TestCase):
    def test_montage(self):
        # basic
        expected = np.ones((2, 2, 3))
        actual = montage(np.ones((4, 1, 1, 3)), image_counts=(2, 2))
        np.testing.assert_array_equal(actual, expected)

        # basic, repeat
        expected = np.ones((2, 2, 3))
        actual = montage(np.ones((1, 1, 1, 3)), image_counts=(2, 2), repeat=True)
        np.testing.assert_array_equal(actual, expected)

        # basic, start 1
        expected = np.ones((2, 2, 3))
        expected[-1, -1] = 0.0
        actual = montage(np.ones((4, 1, 1, 3)), image_counts=(2, 2), start=1)
        np.testing.assert_array_equal(actual, expected)

        # basic, max 1
        expected = np.zeros((2, 2, 3))
        expected[0, 0] = 1.0
        actual = montage(np.ones((4, 1, 1, 3)), image_counts=(2, 2), maximum_images=1)
        np.testing.assert_array_equal(actual, expected)

        # larger
        patch_count = (22, 20)
        patch_shape = tuple(
            [ceil(s / c) for s, c in zip(self.rgb.shape[:-1], patch_count)]
        )
        patches = patchify_image(self.rgb, patch_shape)
        expected = unpatchify_image(patches, image_shape=self.rgb.shape, offset=(0, 0))
        padding = list(
            [
                (0, (p * c) - x)
                for p, c, x in zip(patch_shape, patch_count, self.rgb.shape[:-1])
            ]
        )
        padding.append((0, 0))
        padding = tuple(padding)
        expected = np.pad(expected, padding)
        actual = montage(patches, image_counts=patch_count)
        np.testing.assert_array_equal(actual, expected)

    def test_save_load(self):
        # rgb
        try:
            path = PurePath("image_util_test_output.png")
            expected = self.rgb.astype(np.uint8)
            save(expected, str(path))
            actual = load(str(path))
            np.testing.assert_array_equal(actual, expected)
        finally:
            if Path(path).is_file():
                Path(path).unlink()

        # gray
        try:
            path = PurePath("image_util_test_output.png")
            expected = self.rgb.astype(np.uint8)
            expected = expected[..., 0][..., np.newaxis]
            save(expected, str(path))
            actual = load(str(path))
            np.testing.assert_array_equal(actual, expected)
        finally:
            if Path(path).is_file():
                Path(path).unlink()

        # redundant rgb forced rgb
        try:
            path = PurePath("image_util_test_output.png")
            expected = self.rgb.astype(np.uint8)
            expected[..., 1] = expected[..., 0]
            expected[..., 2] = expected[..., 0]
            save(expected, str(path))
            actual = load(str(path), force_rgb=True)
            np.testing.assert_array_equal(actual, expected)
        finally:
            if Path(path).is_file():
                Path(path).unlink()

        # redundant rgb
        try:
            path = PurePath("image_util_test_output.png")
            expected = self.rgb.astype(np.uint8)
            expected[..., 1] = expected[..., 0]
            expected[..., 2] = expected[..., 0]
            save(expected, str(path))
            actual = load(str(path))
            np.testing.assert_array_equal(actual, expected[..., 0][..., np.newaxis])
        finally:
            if Path(path).is_file():
                Path(path).unlink()

    def test_unpatchify_stack_roundtrip(self):
        OFFSET = (3, 7)
        expected = np.random.rand(*(4, 11, 13, 1))
        actual = patchify_stack(expected, (3, 4), OFFSET)
        actual = unpatchify_stack(actual, expected.shape, OFFSET)
        np.testing.assert_array_equal(actual, expected)

    def test_unpatchify_image(self):
        # 2D, uneven division, offset, roundtrip
        OFFSET = (1, 1)
        expected = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])[..., np.newaxis]
        actual = patchify_image(expected, (2, 3), OFFSET)
        actual = unpatchify_image(actual, expected.shape, OFFSET)
        np.testing.assert_array_equal(actual, expected)

        # 2D, random, roundtrip
        OFFSET = (3, 7)
        expected = np.random.rand(*(11, 13, 1))
        actual = patchify_image(expected, (3, 4), OFFSET)
        actual = unpatchify_image(actual, expected.shape, OFFSET)
        np.testing.assert_array_equal(actual, expected)

        # 2D, photo, roundtrip
        OFFSET = (3, 7)
        expected = self._read_snow_image()
        actual = patchify_image(expected, (32, 32), OFFSET)
        actual = unpatchify_image(actual, expected.shape, OFFSET)
        np.testing.assert_array_equal(actual, expected)

    def test_patchify_image(self):
        # 1D, even division, along X
        im = np.array([0, 1, 2, 3])[..., np.newaxis, np.newaxis]
        actual = patchify_image(im, (2, 1))
        expected = np.array([[0, 1], [2, 3]])[..., np.newaxis, np.newaxis]
        np.testing.assert_array_equal(actual, expected)

        # # 1D, even division, along Y
        im = np.array([[0, 1, 2, 3]])[..., np.newaxis]
        actual = patchify_image(im, (1, 2))
        expected = np.array([[[0, 1]], [[2, 3]]])[..., np.newaxis]
        np.testing.assert_array_equal(actual, expected)

        # # 1D, uneven division, along X
        im = np.array([0, 1, 2, 3, 4])[..., np.newaxis, np.newaxis]
        actual = patchify_image(im, (2, 1))
        expected = np.array([[0, 1], [2, 3], [4, 0]])[..., np.newaxis, np.newaxis]
        np.testing.assert_array_equal(actual, expected)

        # # 1D, uneven division, along Y
        im = np.array([[0, 1, 2, 3, 4]])[..., np.newaxis]
        actual = patchify_image(im, (1, 2))
        expected = np.array([[[0, 1]], [[2, 3]], [[4, 0]]])[..., np.newaxis]
        np.testing.assert_array_equal(actual, expected)

        # # 1D, uneven division, offset, along X
        im = np.array([0, 1, 2, 3, 4])[..., np.newaxis, np.newaxis]
        actual = patchify_image(im, (2, 1), (1, 0))
        expected = np.array([[0, 0], [1, 2], [3, 4]])[..., np.newaxis, np.newaxis]
        np.testing.assert_array_equal(actual, expected)

        # # 1D, uneven division, offset, along Y
        im = np.array([[0, 1, 2, 3, 4]])[..., np.newaxis]
        actual = patchify_image(im, (1, 2), (0, 1))
        expected = np.array([[[0, 0]], [[1, 2]], [[3, 4]]])[..., np.newaxis]
        np.testing.assert_array_equal(actual, expected)

        # 2D, uneven division, offset
        im = np.array([[0, 1, 2], [3, 4, 5]])[..., np.newaxis]
        actual = patchify_image(im, (2, 2), (1, 1))
        expected = np.array(
            [[[0, 0], [0, 0]], [[0, 0], [1, 2]], [[0, 3], [4, 5]], [[0, 0], [0, 0]]]
        )[..., np.newaxis]
        np.testing.assert_array_equal(actual, expected)

        # 1D, channels
        im = np.array([[[0, 1, 2]]])
        actual = patchify_image(im, (2, 2), (1, 1))
        expected = np.array([[[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 1, 2]]]])
        np.testing.assert_array_equal(actual, expected)

    def test_to_dtype(self):
        # unsigned to signed and back
        expected = np.arange(0, 256, dtype=np.uint8)
        im = to_dtype(expected, dtype=np.int8)
        actual = to_dtype(im, dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

        # signed to unsigned and back
        expected = np.arange(-128, 128, dtype=np.int8)
        im = to_dtype(expected, dtype=np.uint8)
        actual = to_dtype(im, dtype=np.int8)
        np.testing.assert_array_equal(actual, expected)

        # float to unsigned and back
        im = np.linspace(0.0, 1.0, 100, dtype=np.float64)
        expected = (im * 255.0).astype(np.uint8).astype(np.float64) / 255.0
        im = to_dtype(im, dtype=np.uint8)
        actual = to_dtype(im, dtype=np.float64)
        np.testing.assert_array_equal(actual, expected)

        # float to signed and back
        im = np.linspace(0.0, 1.0, 100, dtype=np.float64)
        expected = (im * 255.0 - 128.0).astype(np.int8)
        expected = (expected + 128.0).astype(np.float64) / 255.0
        im = to_dtype(im, dtype=np.int8)
        actual = to_dtype(im, dtype=np.float64)
        np.testing.assert_array_equal(actual, expected)

        # unsigned to bool and back
        im = np.arange(0, 256, dtype=np.uint8)
        expected = np.ones_like(im) * 255
        expected[0] = 0
        im = to_dtype(im, dtype=np.bool)
        actual = to_dtype(im, dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

        # signed to logical and back
        im = np.arange(-128, 128, dtype=np.int8)
        expected = np.ones_like(im) * 127
        expected[0] = -128
        im = to_dtype(im, dtype=np.bool)
        actual = to_dtype(im, dtype=np.int8)
        np.testing.assert_array_equal(actual, expected)

        # float to logical and back
        im = np.linspace(0.0, 1.0, 100, dtype=np.float64)
        expected = np.ones_like(im)
        expected[0] = 0.0
        im = to_dtype(im, dtype=np.bool)
        actual = to_dtype(im, dtype=np.float64)
        np.testing.assert_array_equal(actual, expected)

    def setUp(self):
        self.IMAGES = {
            "snow": self._read_snow_image,
            "tulips": self._read_tulips_image,
        }

        self.side_len = np.iinfo(np.uint8).max
        self.base_shape = (self.side_len, self.side_len)

        self.rgb = np.moveaxis(np.indices(self.base_shape), 0, -1).astype(np.uint8)
        self.rgb = np.concatenate(
            (self.rgb, np.zeros(self.base_shape + (1,)).astype(np.uint8)), axis=2
        )
        self.rgb_shape = self.rgb.shape

        self.fov_radius_ratio = 0.45
        self.fov_offset = (-35, 35)
        self.fov_radius = floor(self.side_len * self.fov_radius_ratio)

        self.wait_time = 500

        self.res_path = PurePath("test") / "res"
        self.base_path = self.res_path / "base"
        self.snow_image_path = self.base_path / "snow.jpg"
        self.tulips_image_path = self.base_path / "tulips.jpg"

    def tearDown(self):
        pass

    def show(self, image, tag):
        show(image, tag)
        cv2.moveWindow(tag, 100, 100)
        cv2.waitKey(self.wait_time)
        cv2.destroyWindow(tag)

    def reduce_contrast(self, image):
        factor = 3.0
        minimum = 50
        return (np.round(image / factor) + minimum).astype(np.uint8)

    def _read_gray_image(self):
        out = self._read_tulips_image()
        out = to_gray(out)
        return out

    def _read_snow_image(self):
        return load(str(self.snow_image_path))

    def _read_tulips_image(self):
        return load(str(self.tulips_image_path))

    def run_fn(self, image, fn, name=None, ext=None, *args, **kwargs):
        out = fn(image, *args, **kwargs)
        # vis = np.concatenate((image, out), axis=0)
        # tag = "test: {}".format(fn.__name__)
        # self.show(vis, tag)
        if ext is None:
            ext = ".png"
        if name is not None:
            save(out, self.res_path / (name + ext))
        return out

    def assert_image_equal(self, actual, expected_name, ext=None):
        if ext is None:
            ext = ".png"
        expected = load(self.res_path / (expected_name + ext))
        np.testing.assert_array_equal(actual, expected)

    def standardize(self, image):
        standardized = standardize(image)
        return self.rescale(standardized)

    def rescale(self, image):
        return rescale(image, out_range=(0, 255)).astype(np.uint8)

    def test_adjust_gamma(self):
        base_name = "adjust_gamma_{:s}_{:s}"
        for name, image_fn in self.IMAGES.items():
            name = base_name.format(name, "2")
            out = self.run_fn(image_fn(), adjust_gamma, gamma=2.0, name=name)
            self.assert_image_equal(out, name)
        for name, image_fn in self.IMAGES.items():
            name = base_name.format(name, "05")
            out = self.run_fn(image_fn(), adjust_gamma, gamma=0.5, name=name)
            self.assert_image_equal(out, name)

    def test_clahe(self):
        base_name = "clahe_{:s}_{:s}"
        for name, image_fn in self.IMAGES.items():
            name = base_name.format(name, "default")
            out = self.run_fn(image_fn(), clahe, name=name)
            self.assert_image_equal(out, name)
        for name, image_fn in self.IMAGES.items():
            name = base_name.format(name, "2_2")
            out = self.run_fn(image_fn(), clahe, tile_size=(2, 2), name=name)
            self.assert_image_equal(out, name)
        for name, image_fn in self.IMAGES.items():
            name = base_name.format(name, "20_20")
            out = self.run_fn(image_fn(), clahe, tile_size=(20, 20), name=name)
            self.assert_image_equal(out, name)
        # TODO add structured assertions here

    def test_consensus(self):
        # TWO_CLASS
        A = np.array([[1, 1], [1, 1]])
        B = np.array([[0, 1], [1, 1]])
        C = np.array([[0, 0], [1, 1]])
        D = np.array([[0, 0], [0, 1]])
        data_int = np.stack([A, B, C, D])
        data_int = data_int[..., np.newaxis]
        data_bool = data_int.copy().astype(np.bool)

        self.assertRaises(AssertionError, consensus, data_int[0, ...])
        self.assertRaises(AssertionError, consensus, data_bool[0, ...])
        # self.assertRaises(AssertionError, consensus, data_int, threshold=1)
        self.assertRaises(AssertionError, consensus, data_bool, threshold="majority")

        RESULT_MIN = np.array([[0, 0], [1, 1]])[..., np.newaxis]
        con = consensus(data_bool)
        self.assertTrue((con == RESULT_MIN).all())
        con = consensus(data_int)
        self.assertTrue((con == RESULT_MIN).all())
        con = consensus(data_int, threshold="majority")
        self.assertTrue((con == RESULT_MIN).all())

        RESULT_ZERO = np.array([[1, 1], [1, 1]])[..., np.newaxis]
        con = consensus(data_bool, threshold=0)
        self.assertTrue((con == RESULT_ZERO).all())
        con = consensus(data_bool, threshold=0.0)
        self.assertTrue((con == RESULT_ZERO).all())

        RESULT_ONE = np.array([[0, 1], [1, 1]])[..., np.newaxis]
        con = consensus(data_bool, threshold=1)
        self.assertTrue((con == RESULT_ONE).all())
        con = consensus(data_bool, threshold=0.25)
        self.assertTrue((con == RESULT_ONE).all())

        RESULT_TWO = np.array([[0, 0], [1, 1]])[..., np.newaxis]
        con = consensus(data_bool, threshold=2)
        self.assertTrue((con == RESULT_TWO).all())
        con = consensus(data_bool, threshold=0.5)
        self.assertTrue((con == RESULT_TWO).all())

        RESULT_THREE = np.array([[0, 0], [0, 1]])[..., np.newaxis]
        con = consensus(data_bool, threshold=3)
        self.assertTrue((con == RESULT_THREE).all())
        con = consensus(data_bool, threshold=0.75)
        self.assertTrue((con == RESULT_THREE).all())

        RESULT_FOUR = np.array([[0, 0], [0, 0]])[..., np.newaxis]
        con = consensus(data_bool, threshold=4)
        self.assertTrue((con == RESULT_FOUR).all())
        con = consensus(data_bool, threshold=1.0)
        self.assertTrue((con == RESULT_FOUR).all())

        # MULTI_CLASS
        A = np.array([[1, 2], [2, 2]])
        B = np.array([[0, 1], [2, 2]])
        C = np.array([[0, 1], [1, 2]])
        D = np.array([[0, 0], [1, 1]])
        data_mc = np.stack([A, B, C, D])
        data_mc = data_mc.astype(np.uint8)
        data_mc = data_mc[..., np.newaxis]

        RESULT_MIN = np.array([[0, 1], [1, 2]])[..., np.newaxis]
        con = consensus(data_mc, threshold="majority")
        self.assertTrue((con == RESULT_MIN).all())

    def test_overlay(self):
        bg = self._read_tulips_image()
        fg = self._read_snow_image()
        color = [0.5, 1.0, 0.2]
        self.show(overlay(bg, fg, color, alpha=0.8, beta=0.2), "test: overlay")
        # TODO automate checking

    def test_rescale(self):
        base_name = "rescale_{:s}_{:s}"
        for name, image_fn in self.IMAGES.items():
            name = base_name.format(name, "out_0_128_uint8")
            out = self.run_fn(
                image_fn(),
                rescale,
                out_range=(0, 128),
                dtype=np.uint8,
                clip=True,
                name=name,
            )
            self.assert_image_equal(out, name)
        for name, image_fn in self.IMAGES.items():
            name = base_name.format(name, "in_0_128_float64")
            out = self.run_fn(
                image_fn(),
                rescale,
                in_range=(0, 128),
                dtype=np.uint8,
                clip=True,
                name=name,
            )
            self.assert_image_equal(out, name)

    def test_show(self):
        self.show(self.rgb.astype(np.uint8), "test: visualize_rgb (blue and green?)")
        self.show(self.rgb[..., 0], "test: visualize_gray (is gradient?)")
        self.show(to_dtype(self.mask, np.uint8), "test: visualize_gray (is circle?)")
        self.show(self._read_gray_image(), "test: visualize_gray (is gray?)")
        self.show(self._read_tulips_image(), "test: visualize_color (is color?)")

    def test_stack(self):
        n = 3
        s = stack(n * (self.rgb,))
        self.assertEqual(s.shape[0], n)
        self.assertEqual(s.shape[1:], self.rgb.shape[0:])
        self.assertIsInstance(s, np.ndarray)

    def test_standardize(self):
        self.run_fn(self._read_snow_image(), self.standardize)
        self.run_fn(self._generate_image(), self.standardize)
        # TODO add structured assertions here

    # TODO gray_to_color
    # TODO resize


if __name__ == "__main__":
    unittest.main()
