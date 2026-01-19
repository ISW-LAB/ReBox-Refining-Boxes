#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ultralytics Detect Datasets Bulk Downloader (FINAL)

- Purpose:
  Automatically download user-specified Ultralytics object detection YAML datasets
  and save images + labels.

- Features:
  1) Specify save path via --save-dir in main
  2) Fix download root with settings.update({'datasets_dir': ...})
  3) Sequential download of target_datasets list
  4) Handle YAML naming variations:
     - Automatically try multiple candidate names for each item

Usage:
  python download_ultralytics_detect_list.py --save-dir ./ultralytics_obj/datasets
"""

import os
import argparse
from typing import List, Dict

from ultralytics import settings
from ultralytics.data.utils import check_det_dataset


# ==============================================================================
# Download Directory Configuration
# ==============================================================================
def set_download_directory(target_path: str):
    abs_path = os.path.abspath(target_path)
    os.makedirs(abs_path, exist_ok=True)

    print(f"[INFO] Save folder verified/created: {abs_path}")
    print(f"[INFO] Ultralytics datasets_dir setting: {abs_path}")
    settings.update({'datasets_dir': abs_path})

    print(f"[INFO] Current datasets_dir: {settings['datasets_dir']}")
    print("=" * 70)


# ==============================================================================
# User-defined Download List
# ==============================================================================
def build_target_candidates() -> List[Dict]:
    """
    Expand user-specified target_datasets into candidate list
    considering Ultralytics YAML naming variations.
    """

    # Original user list
    base = [
        # Tiny/Test
        "coco8.yaml",
        "coco128.yaml",

        # General/Benchmark
        "voc.yaml",
        "coco.yaml",
        "lvis.yaml",

        # Specialized
        "VisDrone.yaml",
        "xView.yaml",
        "sku110k.yaml",
        "brain-tumor.yaml",
        "Medical-pills.yaml",
        "KITTI.yaml",
        "african-wildlife.yaml",
        "signature.yaml",
        "HomeObjects-3K.yaml",
        "Construction-PPE.yaml",
    ]

    # Candidate variations to reduce failure rate
    # key: identifier, ymls: try order
    candidates = [
        # {"key": "coco8", "ymls": ["coco8.yaml"]},
        # {"key": "coco128", "ymls": ["coco128.yaml"]},
        # {"key": "voc", "ymls": ["voc.yaml", "VOC.yaml"]},
        # {"key": "coco", "ymls": ["coco.yaml", "COCO.yaml"]},
        # {"key": "lvis", "ymls": ["lvis.yaml", "LVIS.yaml"]},
        {"key": "visdrone", "ymls": ["visdrone.yaml", "VisDrone.yaml"]},
        {"key": "xview", "ymls": ["xview.yaml", "xView.yaml"]},
        # {"key": "sku110k", "ymls": ["sku110k.yaml", "SKU-110K.yaml", "sku-110k.yaml"]},
        # {"key": "brain-tumor", "ymls": ["brain-tumor.yaml"]},
        # {"key": "medical-pills", "ymls": ["Medical-pills.yaml", "medical-pills.yaml"]},
        # {"key": "kitti", "ymls": ["KITTI.yaml", "kitti.yaml"]},
        # {"key": "african-wildlife", "ymls": ["african-wildlife.yaml"]},
        # {"key": "signature", "ymls": ["signature.yaml"]},
        # {"key": "homeobjects-3k", "ymls": ["HomeObjects-3K.yaml", "homeobjects-3k.yaml"]},
        # {"key": "construction-ppe", "ymls": ["Construction-PPE.yaml", "construction-ppe.yaml"]},
    ]

    # Auto-supplement items in base but not in candidates
    cand_keys = {c["key"] for c in candidates}
    for b in base:
        key = b.replace(".yaml", "").lower()
        if key not in cand_keys:
            candidates.append({"key": key, "ymls": [b]})

    return candidates


# ==============================================================================
# Download Execution
# ==============================================================================
def download_from_list(save_dir: str):
    set_download_directory(save_dir)

    targets = build_target_candidates()

    print(f"[INFO] Starting download of {len(targets)} items.\n")

    success = []
    fail = []

    for item in targets:
        key = item["key"]
        ymls = item["ymls"]

        done = False
        last_err = None

        for yml in ymls:
            try:
                print(f"[PROGRESS] {key} <- {yml} ...", end=" ", flush=True)
                info = check_det_dataset(yml)
                _ = info.get("path", None)
                print("[OK]")
                success.append(yml)
                done = True
                break
            except Exception as e:
                print("[FAILED]")
                last_err = e

        if not done:
            print(f"   -> {key} final failure: {type(last_err).__name__}: {last_err}")
            fail.append(key)

    print("\n" + "=" * 70)
    print(f"[RESULT SUMMARY]")
    print(f"  Success: {len(success)}")
    print(f"  Failed: {len(fail)}")
    print(f"\n[SAVE PATH]")
    print(f"  {os.path.abspath(save_dir)}")

    if fail:
        print("\n[FAILED ITEMS]")
        for k in fail:
            print(f"  - {k}")

    print("=" * 70)


# ==============================================================================
# CLI / main
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Download Ultralytics detect datasets from a user-defined YAML list."
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./ultralytics_obj/datasets",
        help="Root folder to save downloaded datasets"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    download_from_list(save_dir=args.save_dir)


if __name__ == "__main__":
    main()
