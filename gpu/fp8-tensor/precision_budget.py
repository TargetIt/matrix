#!/usr/bin/env python3
"""Small analytical FP8 examples, not a model or GPU benchmark.

E4M3 follows the finite encoding in arXiv:2209.05433v2, Table 1.
Only finite inputs are supported; round to nearest, ties to even, saturate.
Per-tensor scaling here uses amax/max_finite for BOTH formats. This is not
an MXFP8 quantizer, vendor runtime, calibration algorithm, or QAT experiment.
MXFP8 storage estimates count 32 values plus one scale byte, without padding.
"""

import json
import math


def e4m3_positive(code):
    if type(code) is not int or not 0 <= code <= 126:
        raise ValueError("Expected a finite nonnegative E4M3 code in [0,126]")
    exponent, fraction = code >> 3, code & 7
    if exponent == 0:
        return fraction * 2.0 ** -9
    return (1 + fraction / 8) * 2.0 ** (exponent - 7)


VALUES = tuple(e4m3_positive(c) for c in range(127))


def round_e4m3(x):
    if not math.isfinite(x):
        raise ValueError("Only finite inputs supported")
    magnitude = min(abs(x), 448.0)
    # Even low significand bit wins an exactly equidistant tie.
    code = min(range(127), key=lambda c: (abs(VALUES[c] - magnitude), c & 1))
    return math.copysign(VALUES[code], x)


def quantize_tensor(values, fmt):
    if fmt not in ("e4m3", "int8") or not values or not all(map(math.isfinite, values)):
        raise ValueError("Use a nonempty finite tensor and e4m3/int8")
    peak = max(map(abs, values))
    if peak == 0:
        return [0.0] * len(values)
    limit = 448.0 if fmt == "e4m3" else 127.0
    scale = peak / limit
    quantizer = round_e4m3 if fmt == "e4m3" else lambda x: max(-127, min(127, round(x)))
    return [scale * quantizer(x / scale) for x in values]


def mse(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b)) / len(a)


def storage(elements):
    if type(elements) is not int or elements < 0:
        raise ValueError("elements must be a nonnegative integer")
    return {"fp16_bytes": 2 * elements, "fp8_payload_bytes": elements,
            "mxfp8_payload_and_scales_bytes": elements + (elements + 31) // 32}


def speedup(neural_fraction, neural_speedup=2.0, extra_overhead_fraction=0.0):
    if not all(map(math.isfinite, (neural_fraction, neural_speedup, extra_overhead_fraction))):
        raise ValueError("Use finite parameters")
    if not 0 <= neural_fraction <= 1 or neural_speedup <= 0 or extra_overhead_fraction < 0:
        raise ValueError("Invalid fraction, speedup, or overhead")
    return 1 / (1 - neural_fraction + neural_fraction / neural_speedup + extra_overhead_fraction)


def report():
    examples = {
        "bounded_uniform": [i / 1000 for i in range(-1000, 1001)],
        "small_values_and_one_outlier": [i / 1000 for i in range(1, 32)] + [64.0],
    }
    errors = {}
    for name, values in examples.items():
        errors[name] = {fmt: {"mse": mse(values, quantize_tensor(values, fmt)),
                             "nonzero_values_rounded_to_zero": sum(x != 0 and y == 0 for x, y in zip(values, quantize_tensor(values, fmt)))}
                        for fmt in ("int8", "e4m3")}
    return {
        "scope": "Synthetic quantization and analytical storage/latency only; no image-quality or GPU results",
        "quantization_examples": errors,
        "activation_480x270x32": storage(480 * 270 * 32),
        "nfru_conv_weights_103232": storage(103232),
        "ideal_32_value_fp16_to_mxfp8_storage_ratio": 64 / 33,
        "pipeline_speedup_with_2x_neural_path": {str(p): speedup(p) for p in (0.25, 0.5, 0.75, 1.0)},
        "pipeline_2ms_neural_2ms_other_plus_0_2ms_conversion_speedup": speedup(0.5, 2.0, 0.05),
    }


if __name__ == "__main__":
    print(json.dumps(report(), indent=2))
