import math
import unittest

from precision_budget import VALUES, e4m3_positive, quantize_tensor, report, round_e4m3, speedup, storage


class PrecisionBudgetTests(unittest.TestCase):
    def test_encoding_boundaries(self):
        self.assertEqual(e4m3_positive(1), 2 ** -9)
        self.assertEqual(e4m3_positive(8), 2 ** -6)
        self.assertEqual(e4m3_positive(126), 448)
        self.assertEqual(len(set(VALUES)), 127)
        self.assertEqual(list(VALUES), sorted(VALUES))

    def test_roundtrip_all_finite_values(self):
        for v in VALUES:
            self.assertEqual(round_e4m3(v), v)
            self.assertEqual(round_e4m3(-v), -v)

    def test_ties_saturation_and_signed_zero(self):
        self.assertEqual(round_e4m3(1.0625), 1.0)
        self.assertEqual(round_e4m3(1.1875), 1.25)
        self.assertEqual(round_e4m3(999), 448)
        self.assertEqual(round_e4m3(-999), -448)
        self.assertEqual(math.copysign(1, round_e4m3(-0.0)), -1)

    def test_distribution_dependent_error(self):
        examples = report()["quantization_examples"]
        self.assertLess(examples["bounded_uniform"]["int8"]["mse"], examples["bounded_uniform"]["e4m3"]["mse"])
        self.assertLess(examples["small_values_and_one_outlier"]["e4m3"]["mse"], examples["small_values_and_one_outlier"]["int8"]["mse"])
        self.assertEqual(quantize_tensor([0, 0], "e4m3"), [0, 0])

    def test_storage_and_amdahl(self):
        self.assertEqual(storage(32)["mxfp8_payload_and_scales_bytes"], 33)
        self.assertEqual(storage(33)["mxfp8_payload_and_scales_bytes"], 35)
        self.assertEqual(storage(0)["mxfp8_payload_and_scales_bytes"], 0)
        self.assertAlmostEqual(speedup(0.5), 4 / 3)
        self.assertAlmostEqual(speedup(0.5, 2, 0.05), 1.25)
        self.assertEqual(speedup(1), 2)

    def test_invalid_inputs(self):
        for v in (float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                round_e4m3(v)
        with self.assertRaises(ValueError):
            e4m3_positive(127)
        with self.assertRaises(ValueError):
            quantize_tensor([], "int8")
        with self.assertRaises(ValueError):
            speedup(1.1)


if __name__ == "__main__":
    unittest.main()
