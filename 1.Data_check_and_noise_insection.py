#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_det_inspect_then_noise.py

1) Inspect detection datasets under /datasets using ultra_det_loader
2) Select the first spec with train images
3) DataLoader batch sanity check
4) Generate noisy labels using noisy_insection:
   - labels_uniform_scaling_{S}
   - labels_boundary_jitter_{K}
   saved at the same level as labels/
5) For each dataset, randomly sample n=3 from train and save to
   /datasets/_noise_reports/noise_check/:
     - {dataset}__n{i}__original.jpg
     - {dataset}__n{i}__noisecase_{labels_uniform_scaling_S}.jpg
     - {dataset}__n{i}__noisecase_{labels_boundary_jitter_K}.jpg
   (no per-dataset subfolder)

Note:
- noisy_insection.py, ultra_det_loader.py must be in PROJECT_MODULE_DIR
"""

import sys
import json
import random
import inspect as _inspect
from pathlib import Path
from typing import List, Tuple, Optional

from PIL import Image, ImageDraw, ImageFont


# -----------------------------------------------------------------------------
# 0) Module path registration
# -----------------------------------------------------------------------------
PROJECT_MODULE_DIR = Path("/home/ISW/project/Project_Module")
if str(PROJECT_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_MODULE_DIR))


# -----------------------------------------------------------------------------
# 1) ultra_det_loader import
# -----------------------------------------------------------------------------
from ultra_det_loader import (
    inspect_det_datasets,
    build_dataset,
    build_dataloader,
    count_split_images,
)


# -----------------------------------------------------------------------------
# 2) noisy_insection import
# -----------------------------------------------------------------------------
try:
    from noisy_insection import (
        generate_noisy_labels,
        generate_uniform_scaling_noise_for_dataset,
        generate_boundary_jitter_noise_for_dataset,
        UNIFORM_SCALING_FACTORS,
        JITTER_PATTERNS,
        SEED as NOISE_SEED_DEFAULT,
        read_yolo_labels,   # Already implemented in noisy_insection
    )
except Exception as e:
    raise ImportError(
        "noisy_insection.py must be placed in PROJECT_MODULE_DIR and importable."
    ) from e


# -----------------------------------------------------------------------------
# User config
# -----------------------------------------------------------------------------
load_dir = "/home/ISW/project/datasets"

# ultra_det_loader inspect options
VISUALIZE = True   # False for stats only
IMG_SIZE = 640
N_SAMPLES = 3

# noise generation options
NOISE_MODE = "both"        # "isotropic" | "borderwise" | "both"
NOISE_SEED = 42
OVERWRITE_NOISE = False

# True: generate for all datasets under load_dir
# False: generate only for "selected spec.root"
GENERATE_FOR_ALL_DATASETS = True

# Report saving (only if module supports return_report)
SAVE_REPORT_JSON = True

# Noise check visualization settings
SAVE_NOISE_CHECK = True
# Save to _noise_reports/noise_check
NOISE_CHECK_DIR = Path(load_dir) / "_noise_reports" / "noise_check"
NOISE_CHECK_RANDOM_SEED = 123
NOISE_CHECK_N_PER_DATASET = 3


# -----------------------------------------------------------------------------
# Helper: List of all noisecase folder names for visualization
# -----------------------------------------------------------------------------
def get_all_visual_noise_cases() -> List[str]:
    """
    Returns e.g.:
      ["labels_uniform_scaling_0.7", "labels_uniform_scaling_0.8", ...,
       "labels_boundary_jitter_3", "labels_boundary_jitter_4", ...]
    """
    cases: List[str] = []
    if NOISE_MODE in ["isotropic", "both"]:
        for s in UNIFORM_SCALING_FACTORS:
            cases.append(f"labels_uniform_scaling_{s}")
    if NOISE_MODE in ["borderwise", "both"]:
        for k in JITTER_PATTERNS:
            cases.append(f"labels_boundary_jitter_{k}")
    return cases


# -----------------------------------------------------------------------------
# Helper: image -> label path mapping (labels dir name configurable)
# -----------------------------------------------------------------------------
def image_to_label_path(img_path: Path, ds_root: Path, labels_dir_name: str = "labels") -> Path:
    """
    .../images/.../xxx.jpg -> .../{labels_dir_name}/.../xxx.txt
    """
    s = str(img_path)

    # Most common case
    if "/images/" in s:
        s2 = s.replace("/images/", f"/{labels_dir_name}/")
        return Path(s2).with_suffix(".txt")

    # fallback
    return ds_root / labels_dir_name / (img_path.stem + ".txt")


# -----------------------------------------------------------------------------
# Helper: YOLO norm -> abs xyxy
# -----------------------------------------------------------------------------
def yolo_norm_to_xyxy_abs(
    cx: float, cy: float, w: float, h: float,
    img_w: int, img_h: int
) -> Tuple[float, float, float, float]:
    bw = w * img_w
    bh = h * img_h
    x1 = (cx * img_w) - bw / 2.0
    y1 = (cy * img_h) - bh / 2.0
    x2 = x1 + bw
    y2 = y1 + bh

    x1 = max(0.0, min(float(img_w), x1))
    y1 = max(0.0, min(float(img_h), y1))
    x2 = max(0.0, min(float(img_w), x2))
    y2 = max(0.0, min(float(img_h), y2))
    return x1, y1, x2, y2


# -----------------------------------------------------------------------------
# Helper: draw boxes overlay (PIL)
# -----------------------------------------------------------------------------
def draw_boxes(
    img: Image.Image,
    labels: List[Tuple[int, float, float, float, float]],
    class_names: Optional[List[str]] = None,
) -> Image.Image:
    img = img.copy()
    draw = ImageDraw.Draw(img)

    font = None
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    W, H = img.size

    for cls, cx, cy, w, h in labels:
        x1, y1, x2, y2 = yolo_norm_to_xyxy_abs(cx, cy, w, h, W, H)
        if x2 <= x1 or y2 <= y1:
            continue

        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)

        name = str(cls)
        if class_names and 0 <= cls < len(class_names):
            name = class_names[cls]

        text = name
        try:
            if font is not None:
                bbox = draw.textbbox((0, 0), text, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            else:
                tw, th = len(text) * 6, 10
        except Exception:
            tw, th = len(text) * 6, 10

        bx1, by1 = x1, max(0, y1 - th - 2)
        bx2, by2 = x1 + tw + 4, y1
        draw.rectangle([bx1, by1, bx2, by2], fill=(255, 0, 0))
        draw.text((bx1 + 2, by1 + 1), text, fill=(255, 255, 255), font=font)

    return img


# -----------------------------------------------------------------------------
# Helper: per-dataset random sample saving
# -----------------------------------------------------------------------------
def save_noise_check_for_all_specs(specs):
    """
    Visualize and save:
       - All uniform scaling cases
       - All boundary jitter cases
    """
    if not SAVE_NOISE_CHECK:
        return

    NOISE_CHECK_DIR.mkdir(parents=True, exist_ok=True)

    vis_cases = get_all_visual_noise_cases()
    rnd = random.Random(NOISE_CHECK_RANDOM_SEED)

    print("\n" + "=" * 80)
    print("[Noise check visualization]")
    print(f"Save dir: {NOISE_CHECK_DIR}")
    print(f"Visual noise cases: {vis_cases if vis_cases else '(none)'}")
    print(f"Samples per dataset: {NOISE_CHECK_N_PER_DATASET}")
    print("=" * 80)

    for spec in specs:
        # Check if train images exist
        try:
            ds = build_dataset(spec, split="train", img_size=IMG_SIZE, strict_exists=False)
        except Exception:
            continue

        if len(ds) == 0:
            continue

        imgs = getattr(ds, "images", None)
        if not imgs:
            continue

        # Select random n samples
        n = min(NOISE_CHECK_N_PER_DATASET, len(imgs))
        chosen = rnd.sample(list(imgs), k=n) if len(imgs) >= n else list(imgs)

        # class names (use if available)
        class_names = getattr(ds, "class_names", None)

        saved_any = False

        for i, img_path in enumerate(chosen, start=1):
            img_path = Path(img_path)

            # ---------- original ----------
            orig_label_path = image_to_label_path(img_path, spec.root, "labels")
            orig_labels = read_yolo_labels(orig_label_path)

            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                continue

            img_orig = draw_boxes(img, orig_labels, class_names=class_names)
            out_orig = NOISE_CHECK_DIR / f"{spec.name}__n{i}__original.jpg"
            try:
                img_orig.save(out_orig)
                saved_any = True
            except Exception:
                pass

            # ---------- all noisecases ----------
            for case_dir in vis_cases:
                noise_label_path = image_to_label_path(img_path, spec.root, case_dir)
                if not noise_label_path.exists():
                    # Skip if noise folder not yet created
                    # (some datasets may not have this case)
                    continue

                noise_labels = read_yolo_labels(noise_label_path)
                img_noise = draw_boxes(img, noise_labels, class_names=class_names)

                out_noise = NOISE_CHECK_DIR / f"{spec.name}__n{i}__noisecase_{case_dir}.jpg"
                try:
                    img_noise.save(out_noise)
                    saved_any = True
                except Exception:
                    pass

        if saved_any:
            print(f"[OK] Saved noise-check samples: {spec.name} (n={n})")
        else:
            print(f"[SKIP] Image/label read issue: {spec.name}")


# -----------------------------------------------------------------------------
# 3) Full inspection (count/classes/sample saving)
# -----------------------------------------------------------------------------
specs = inspect_det_datasets(
    load_dir=load_dir,
    img_size=IMG_SIZE,
    visualize=VISUALIZE,
    n_samples=N_SAMPLES,
)

# -----------------------------------------------------------------------------
# 4) Select first spec with train images
# -----------------------------------------------------------------------------
spec = next((s for s in specs if count_split_images(s, "train") > 0), None)
if spec is None:
    raise RuntimeError("No detection dataset found with at least 1 train image.")

print(f"\n[Selected spec] {spec.name} | root={spec.root} | yaml={spec.yaml_path}")

# -----------------------------------------------------------------------------
# 5) Load train split
# -----------------------------------------------------------------------------
train_ds = build_dataset(spec, split="train", img_size=IMG_SIZE, strict_exists=True)

# -----------------------------------------------------------------------------
# 6) Create DataLoader
# -----------------------------------------------------------------------------
train_loader = build_dataloader(train_ds, batch_size=4, num_workers=4)

# -----------------------------------------------------------------------------
# 7) Batch sanity check
# -----------------------------------------------------------------------------
images, targets = next(iter(train_loader))
print("\n[Batch sanity check]")
print("num_images:", len(images))
print("target keys:", targets[0].keys())
print("boxes shape:", targets[0]["boxes"].shape)
print("labels shape:", targets[0]["labels"].shape)
print("img_path:", targets[0]["img_path"])
print("label_path:", targets[0]["label_path"])


# -----------------------------------------------------------------------------
# 8) Noise generation execution
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("[Noise generation]")

if GENERATE_FOR_ALL_DATASETS:
    sig = _inspect.signature(generate_noisy_labels)
    supports_report = "return_report" in sig.parameters

    kwargs = dict(
        load_dir=load_dir,
        mode=NOISE_MODE,
        uniform_scaling_factors=UNIFORM_SCALING_FACTORS,
        jitter_patterns=JITTER_PATTERNS,
        seed=NOISE_SEED,
        overwrite=OVERWRITE_NOISE,
        verbose=True,
    )

    if supports_report:
        report = generate_noisy_labels(**kwargs, return_report=True)
    else:
        report = None
        generate_noisy_labels(**kwargs)

    # (Optional) Save report
    if SAVE_REPORT_JSON and report is not None:
        out_dir = Path(load_dir) / "_noise_reports"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "noisy_insection_report.json"

        if hasattr(report, "to_dict"):
            payload = report.to_dict()
        else:
            payload = {
                "load_dir": load_dir,
                "mode": NOISE_MODE,
                "seed": NOISE_SEED,
                "uniform_scaling_factors": UNIFORM_SCALING_FACTORS,
                "jitter_patterns": JITTER_PATTERNS,
            }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        print(f"[Report saved] {out_path}")

else:
    ds_root = Path(spec.root)

    print(f"Target dataset root: {ds_root}")

    if NOISE_MODE in ["isotropic", "both"]:
        print(f" - uniform scaling cases: {UNIFORM_SCALING_FACTORS}")
        generate_uniform_scaling_noise_for_dataset(
            ds_root=ds_root,
            uniform_scaling_factors=UNIFORM_SCALING_FACTORS,
            overwrite=OVERWRITE_NOISE,
        )

    if NOISE_MODE in ["borderwise", "both"]:
        print(f" - boundary jitter cases: {JITTER_PATTERNS}")
        generate_boundary_jitter_noise_for_dataset(
            ds_root=ds_root,
            jitter_patterns=JITTER_PATTERNS,
            seed=NOISE_SEED,
            overwrite=OVERWRITE_NOISE,
        )

print("\n[OK] Noise label generation done.")
print("=" * 80)


# -----------------------------------------------------------------------------
# 9) Save original vs all noisecase random 3 samples per dataset
# -----------------------------------------------------------------------------
save_noise_check_for_all_specs(specs)

print("\n[OK] All done.")
print("=" * 80)
