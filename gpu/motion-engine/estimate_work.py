#!/usr/bin/env python3
"""Analytical NFRU workload estimate, NOT a hardware benchmark.

Source: arm/neural-graphics-model-gym commit
b86ee99125ea01c9ec1acf471743e5eb2478414a (2026-09-08).
BlockMatch: levels=6, last_bm_level=2, radius=3, template=5, hints=True.
CNN: count dense convolution MACs; omit bias, BN, activation, shaders,
resampling, quantization, optical-flow refinement and memory operations.
One MAC means one multiply-accumulate (two arithmetic operations).
Python standard library only; prints results without writing files.
"""

import argparse
import json


# name, input channels, output channels, kernel, spatial scale from CNN input
LAYERS = (
    ("conv1", 16, 32, 3, 1),
    ("conv2", 32, 16, 5, 1),
    ("conv3", 16, 16, 3, 1),
    ("skip1_conv", 16, 16, 3, 1),
    ("conv5", 16, 16, 5, 2),
    ("conv5a", 16, 16, 1, 2),
    ("conv5b", 16, 16, 3, 2),
    ("conv5c", 16, 16, 7, 2),
    ("conv5c_1", 16, 16, 7, 2),
    ("conv5d", 16, 16, 7, 2),
    ("conv5d_1", 16, 16, 7, 2),
    ("conv5d_2", 16, 16, 7, 2),
    ("conv5e", 64, 16, 1, 2),
    ("conv6", 16, 16, 3, 1),
    ("conv7", 32, 16, 3, 1),
    ("output_conv_mv", 16, 4, 5, 1),
)


def estimate(width, height, base_fps=30, generated_per_pair=1):
    # Require /8 so CNN downsample/upsample and skip shapes match exactly.
    if width <= 0 or height <= 0 or width % 8 or height % 8:
        raise ValueError("Use positive full-color dimensions divisible by 8.")
    if base_fps <= 0 or generated_per_pair < 1:
        raise ValueError("base_fps must be positive and generated_per_pair >= 1")
    pyramid = []
    h, w = height, width
    for level in range(6):
        padded_h, padded_w = h + h % 2, w + w % 2
        pyramid.append((level, padded_w, padded_h))
        w, h = padded_w // 2, padded_h // 2
    stages = []
    for level, w, h in reversed(pyramid[2:]):
        candidates = 49 + (level == 2)  # additional hint at finest matched level
        stages.append({"level": level, "width": w, "height": h,
                       "candidates_per_position": candidates,
                       "sad_terms": w * h * candidates * 25})
    cnn_width, cnn_height = width // 4, height // 4
    layer_rows = []
    for name, cin, cout, k, scale in LAYERS:
        out_w, out_h = cnn_width // scale, cnn_height // scale
        weights = cin * cout * k * k
        layer_rows.append({"name": name, "width": out_w, "height": out_h,
                           "weights": weights, "macs": out_w * out_h * weights})
    macs = sum(x["macs"] for x in layer_rows)
    sad_terms = sum(x["sad_terms"] for x in stages)
    weights = sum(x["weights"] for x in layer_rows)
    biases = sum(x[2] for x in LAYERS)
    bn_parameters = 2 * sum(x[2] for x in LAYERS[:-1])
    pixels = cnn_width * cnn_height
    return {
        "source_commit": "b86ee99125ea01c9ec1acf471743e5eb2478414a",
        "full_color_size": [width, height],
        "cnn_input_size": [cnn_width, cnn_height],
        "base_fps": base_fps,
        "generated_per_pair": generated_per_pair,
        "blockmatch_stages_coarse_to_fine": stages,
        "sad_terms_per_pair": sad_terms,
        "sad_terms_per_second": sad_terms * base_fps,
        "cnn_layers": layer_rows,
        "cnn_macs_per_generated_frame": macs,
        "cnn_macs_per_second": macs * base_fps * generated_per_pair,
        "cnn_weights": weights,
        "cnn_trainable_parameters_with_bias_bn": weights + biases + bn_parameters,
        "hypothetical_materialized_50_candidate_patches_uint8_bytes": pixels * 50 * 25,
        "hypothetical_materialized_50_candidate_costs_int32_bytes": pixels * 50 * 4,
        "cnn_input_int8_bytes": pixels * 16,
        "cnn_conv1_output_int8_bytes": pixels * 32,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--base-fps", type=float, default=30)
    parser.add_argument("--generated-per-pair", type=int, default=1)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = estimate(args.width, args.height, args.base_fps, args.generated_per_pair)
    except ValueError as exc:
        parser.error(str(exc))
    if args.json:
        print(json.dumps(result, indent=2))
        return
    print(f"Full-color input: {result['full_color_size']}; CNN: {result['cnn_input_size']}")
    for row in result["blockmatch_stages_coarse_to_fine"]:
        print(f"L{row['level']}: {row['width']}x{row['height']}, "
              f"{row['candidates_per_position']} candidates, {row['sad_terms']:,} SAD terms")
    print(f"SAD terms/pair: {result['sad_terms_per_pair']:,}")
    print(f"CNN MACs/generated frame: {result['cnn_macs_per_generated_frame']:,}")
    print(f"CNN weights: {result['cnn_weights']:,}; trainable including bias/BN: "
          f"{result['cnn_trainable_parameters_with_bias_bn']:,}")
    print(f"SAD terms/s: {result['sad_terms_per_second']:,.0f}; "
          f"CNN MACs/s: {result['cnn_macs_per_second']:,.0f}")
    print("Analytical operation counts only; no latency, power or hardware throughput measured.")


if __name__ == "__main__":
    main()
