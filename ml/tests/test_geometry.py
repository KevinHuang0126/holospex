import unittest

from holospex_ml.geometry import ImageSize, ResizeTransform


class GeometryTests(unittest.TestCase):
    def test_letterbox_inverse_accounts_for_padding(self) -> None:
        transform = ResizeTransform.letterbox(ImageSize(1920, 1080), ImageSize(640, 640))
        self.assertEqual(transform.resized, ImageSize(640, 360))
        self.assertEqual((transform.pad_left, transform.pad_top), (0, 140))
        self.assertEqual(transform.to_original((0, 140)), (0, 0))
        self.assertEqual(transform.to_original((640, 500)), (1920, 1080))

    def test_integer_rounding_roundtrips_image_edges(self) -> None:
        transform = ResizeTransform.letterbox(ImageSize(853, 479), ImageSize(640, 640))
        self.assertNotEqual(transform.scale_x, transform.scale_y)
        for point in [(0, 0), (853, 479), (211.5, 43.25)]:
            restored = transform.to_original(transform.to_processed(point))
            self.assertAlmostEqual(restored[0], point[0])
            self.assertAlmostEqual(restored[1], point[1])

    def test_stretch_uses_independent_axes(self) -> None:
        transform = ResizeTransform.stretch(ImageSize(1920, 1080), ImageSize(640, 640))
        self.assertEqual(transform.to_original((320, 320)), (960, 540))

    def test_padding_is_not_silently_clamped(self) -> None:
        transform = ResizeTransform.letterbox(ImageSize(1920, 1080), ImageSize(640, 640))
        self.assertLess(transform.to_original((10, 0))[1], 0)

    def test_invalid_geometry_rejected(self) -> None:
        for width, height in [(0, 10), (10, -1), (True, 10), (10.5, 10)]:
            with self.assertRaises(ValueError):
                ImageSize(width, height)
        transform = ResizeTransform.stretch(ImageSize(100, 100), ImageSize(50, 50))
        with self.assertRaises(ValueError):
            transform.to_original((float("nan"), 0))
        with self.assertRaises(ValueError):
            transform.polygon_to_original([(0, 0), (1, 1)])


if __name__ == "__main__":
    unittest.main()
