"""Regression checks for report arithmetic and scope, not a GPU benchmark."""

import unittest

from estimate_work import estimate


class WorkEstimateTests(unittest.TestCase):
    def test_1080p_pyramid_padding_and_counts(self):
        result = estimate(1920, 1080)
        stages = result["blockmatch_stages_coarse_to_fine"]
        self.assertEqual([(s["width"], s["height"]) for s in stages],
                         [(60, 34), (120, 68), (240, 136), (480, 270)])
        self.assertEqual(result["sad_terms_per_pair"], 214_479_000)
        self.assertEqual(result["sad_reduction_adds_per_pair"], 205_899_840)
        self.assertEqual(result["minimum_selection_comparisons_per_pair"], 8_406_720)
        self.assertEqual(result["cnn_macs_per_generated_frame"], 6_312_038_400)
        self.assertEqual(result["cnn_trainable_parameters_with_bias_bn"], 104_004)

    def test_search_and_hint_sensitivity(self):
        self.assertEqual(estimate(1920, 1080, search_radius=1)["sad_terms_per_pair"], 42_039_000)
        self.assertEqual(estimate(1920, 1080, search_radius=2)["sad_terms_per_pair"], 111_015_000)
        base = estimate(1920, 1080)
        no_hint = estimate(1920, 1080, hints=False)
        self.assertEqual(base["sad_terms_per_pair"] - no_hint["sad_terms_per_pair"], 480 * 270 * 25)
        self.assertEqual(base["cnn_macs_per_generated_frame"], no_hint["cnn_macs_per_generated_frame"])

    def test_flow_reused_across_generated_frames(self):
        one = estimate(1920, 1080)
        three = estimate(1920, 1080, generated_per_pair=3)
        self.assertEqual(one["sad_terms_per_second"], three["sad_terms_per_second"])
        self.assertEqual(one["cnn_macs_per_second"] * 3, three["cnn_macs_per_second"])

    def test_uncapped_reference_resolution(self):
        for w, h in [(1280, 720), (1920, 1080), (2560, 1440)]:
            self.assertEqual(estimate(w, h)["cnn_macs_per_generated_frame"], w * h // 16 * 48_704)
        self.assertGreater(estimate(2560, 1440)["sad_terms_per_pair"], estimate(1920, 1080)["sad_terms_per_pair"])

    def test_invalid_inputs(self):
        for kwargs in ({"width": 1921}, {"height": 0}, {"width": 1920.0},
                       {"base_fps": float("nan")}, {"base_fps": float("inf")},
                       {"generated_per_pair": 1.5}, {"search_radius": 4}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                estimate(**({"width": 1920, "height": 1080} | kwargs))


if __name__ == "__main__":
    unittest.main()
