"""
Exports YOLOv8 models to NCNN format optimized for ARM (Raspberry Pi).

Run on the machine (faster), then copy the generated directories
to the `models/` directory on the Raspberry Pi.

Usage:
    python export_model.py                           # exports only the base model
    python export_model.py --ppe models/ppe_detector.pt  # exports also the PPE model

Supported formats:
    - ncnn   (recommended for ARM, uses NEON SIMD)
    - tflite (alternative, requires tflite-runtime)
    - onnx   (universal fallback)
"""

import argparse
import os
import sys

from ultralytics import YOLO


def export_model(model_path: str, fmt: str) -> str:
    """Exports a YOLO model to the given format. Returns the output path."""
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        sys.exit(1)

    print(f"[EXPORT] Loading {model_path}...")
    model = YOLO(model_path)

    print(f"[EXPORT] Exporting to format '{fmt}'...")
    output_path = model.export(format=fmt)
    print(f"[EXPORT] Model exported: {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export YOLOv8 models for Raspberry Pi (ARM)",
    )
    parser.add_argument(
        "--base",
        default="models/yolov8n.pt",
        help="Path to base YOLOv8 model (default: models/yolov8n.pt)",
    )
    parser.add_argument(
        "--ppe",
        default=None,
        help="Path to fine-tuned PPE model (optional)",
    )
    parser.add_argument(
        "--format",
        choices=["ncnn", "tflite", "onnx"],
        default="ncnn",
        help="Export format (default: ncnn)",
    )
    args = parser.parse_args()

    print(f"[EXPORT] Target format: {args.format}")
    print(f"[EXPORT] Base model: {args.base}")

    export_model(args.base, args.format)

    if args.ppe:
        print(f"\n[EXPORT] PPE model: {args.ppe}")
        export_model(args.ppe, args.format)

    print("\n[EXPORT] Export completed.")
    print("[EXPORT] Copy the generated directories to the models/ directory on the RPi.")


if __name__ == "__main__":
    main()
