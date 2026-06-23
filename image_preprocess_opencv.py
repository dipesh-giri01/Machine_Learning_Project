"""
OpenCV-based image preprocessing for Python (grayscale, aspect ratio, intensity outliers).

Common practice (see e.g. PyImageSearch resize guides, microscopy percentile normalization docs):
    - Preserve aspect ratio when fitting a fixed CNN input size using letterboxing (resize
      so the image fits inside the target box, pad the rest).
    - Handle outlier *pixels* with percentile clipping / robust normalization: a few dead
      hot pixels or glare should not dictate min-max scaling across the whole image.

This module avoids torch/PIL so you can use it standalone or wrap outputs in ndarray → tensor.

ExamCheatingDataset figures below are copied from saved outputs of `exam_cheating_eda.ipynb`
(so they stay aligned with what you plotted in EDA).

References (general preprocessing): PyImageSearch `cv2.resize` / letterbox patterns; robust
percentile clipping for outlier pixels (common in microscopy and ML normalization write-ups).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple, Literal, Union, Optional, TypedDict, Iterable, List

import argparse
import cv2
import numpy as np

ResizeMode = Literal["letterbox", "stretch"]

# Train/test layouts from EDA; used when batch-scanning folders.
DEFAULT_IMAGE_EXTS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}
)

# --- EDA-derived constants (exam_cheating_eda.ipynb, ExamCheatingDataset) -------------------


class ExamCheatingEDA(TypedDict, total=False):
    """Structured summary keyed to the EDA notebook; optional fields extend over time."""

    notebook: str
    dataset_folder: str
    total_images: int
    train_images: int
    test_images: int
    n_classes: int
    classes: Tuple[str, ...]
    imbalance_ratio_train: float
    train_pct: float
    train_width_px: Tuple[int, int, float]  # min, max, mean
    train_height_px: Tuple[int, int, float]
    modes_rgb_vs_rgba: Tuple[int, int]  # count RGB, count RGBA in train manifest
    unique_resolution_combos_train: int
    aspect_ratio_train_describe: dict  # keys: mean, std, min, p25, p50, p75, max


EXAM_CHEATING_EDA: ExamCheatingEDA = {
    "notebook": "exam_cheating_eda.ipynb",
    "dataset_folder": "ExamCheatingDataset",
    "total_images": 2058,
    "train_images": 1562,
    "test_images": 496,
    "n_classes": 5,
    "classes": (
        "cheating",
        "giving code",
        "giving object",
        "looking friend",
        "normal act",
    ),
    # max_count / min_count on train (~640 / ~12, ≈53.33× imbalance in EDA)
    "imbalance_ratio_train": 53.33,
    "train_pct": 1562 / 2058 * 100.0,
    "train_width_px": (72, 1047, 291.75672215108834),
    "train_height_px": (68, 1500, 263.66581306017925),
    "modes_rgb_vs_rgba": (1508, 54),
    "unique_resolution_combos_train": 1096,
    "aspect_ratio_train_describe": {
        "mean": 1.10,
        "std": 0.26,
        "min": 0.37,
        "p25": 0.97,
        "p50": 1.08,
        "p75": 1.23,
        "max": 1.91,
    },
}


def exam_cheating_preprocess_defaults() -> dict:
    """
    Defaults informed by `exam_cheating_eda.ipynb` resolution & aspect-ratio stats:

    - Median dims ≈298×270; letterboxing to ``(299, 299)`` matches typical pretrained backbones.
    - Rare RGBA files (≈54 in train manifest): composite onto white before greyscale/BGR ops.
    - Intensity outliers: per-image percentile clip (see ``outlier_percentiles``).

    Modeling note from EDA: strong class imbalance (~53×); combine with augmentation or weighted
    sampling in your trainer — not handled here.
    """
    return {
        "target_hw": (299, 299),
        "grayscale": True,
        "resize_mode": "letterbox",
        "pad_value": 0,
        "outlier_percentiles": (2.0, 99.0),
        "normalize_01": True,
        # Slight slack beyond observed extremes (1047 × 1500) to guard corrupt headers only.
        "sanity_dimension_min_side": 60,
        "sanity_dimension_max_side": 1600,
    }


def exam_cheating_dimensions_plausible(height: int, width: int) -> bool:
    """Return False for absurd sizes (beyond what EDA saw + small slack)."""
    d = exam_cheating_preprocess_defaults()
    return filter_extreme_dimensions(
        (height, width),
        max_side=int(d["sanity_dimension_max_side"]),
        min_side=int(d["sanity_dimension_min_side"]),
    )


def preprocess_exam_cheating_path(path: Union[str, Path]) -> np.ndarray:
    """Convenience: load + preprocess using :func:`exam_cheating_preprocess_defaults`."""
    d = exam_cheating_preprocess_defaults()
    d_pop = dict(d)
    sanity_min = int(d_pop.pop("sanity_dimension_min_side"))
    sanity_max = int(d_pop.pop("sanity_dimension_max_side"))

    img = imread_preprocessable_bgr(path)
    h, w = img.shape[:2]
    if not filter_extreme_dimensions((h, w), max_side=sanity_max, min_side=sanity_min):
        raise ValueError(
            f"Image dimensions {(h, w)} outside EDA-aligned sanity bounds "
            f"(min_side={sanity_min}, max_side={sanity_max}). See EXAM_CHEATING_EDA."
        )
    return preprocess_image(img, **d_pop)


def bgra_to_bgr_composite_white(bgra: np.ndarray) -> np.ndarray:
    """Composite RGBA/BGRA onto white (matches common handling for transparent screenshots)."""
    if bgra.ndim != 3 or bgra.shape[2] != 4:
        raise ValueError(f"Expected HxWx4 BGRA, got {bgra.shape}")
    alpha = bgra[:, :, 3:4].astype(np.float32) / 255.0
    bgr = bgra[:, :, :3].astype(np.float32)
    bg = np.float32([255.0, 255.0, 255.0])
    blended = bgr * alpha + bg.reshape(1, 1, 3) * (1.0 - alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)


def imread_preprocessable_bgr(path: Union[str, Path]) -> np.ndarray:
    """
    Read an image path into BGR uint8, handling grayscale and RGBA (≈54 train RGBA rows in EDA).

    Uses ``IMREAD_UNCHANGED`` then normalizes layout so :func:`preprocess_image` always gets HxWx3.
    """
    p = Path(path)
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Unable to read image: {p}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = bgra_to_bgr_composite_white(img)
    elif img.shape[2] != 3:
        raise ValueError(f"Unsupported channel count ({img.shape[2]}) for {p}")
    return img


def bgr_to_grayscale(bgr: np.ndarray) -> np.ndarray:
    """BGR uint8/float → single-channel grayscale (same dtype as input where possible)."""
    if bgr.ndim == 2:
        return bgr
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 BGR image, got shape {bgr.shape}")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return gray


def letterbox_resize(
    img: np.ndarray,
    target_h: int,
    target_w: int,
    pad_value: Union[int, float] = 0,
    interp_down: int = cv2.INTER_AREA,
    interp_up: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """
    Resize so the entire image fits in (target_h, target_w) while keeping aspect ratio,
    pad symmetrically with `pad_value` (typically 0 for black borders).
    """
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid image shape {img.shape}")

    scale = min(target_h / h, target_w / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    interp = interp_down if scale < 1.0 else interp_up
    resized = cv2.resize(img, (nw, nh), interpolation=interp)

    pad_h = target_h - nh
    pad_w = target_w - nw
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2

    if img.ndim == 2:
        out = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=float(pad_value),
        )
    else:
        out = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(
                pad_value if isinstance(pad_value, (list, tuple, np.ndarray))
                else [pad_value, pad_value, pad_value]
            ),
        )
    return out.astype(img.dtype, copy=False)


def stretch_resize(
    img: np.ndarray,
    target_h: int,
    target_w: int,
    interp_down: int = cv2.INTER_AREA,
    interp_up: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Classic resize — may distort aspect ratio (not recommended for geometry-sensitive tasks)."""
    h, w = img.shape[:2]
    interp = interp_down if (target_h < h or target_w < w) else interp_up
    return cv2.resize(img, (target_w, target_h), interpolation=interp)


def clip_intensity_percentiles(
    img: np.ndarray,
    low_pct: float = 2.0,
    high_pct: float = 99.0,
) -> np.ndarray:
    """
    Clip pixel intensities to [low_pct, high_pct] percentiles **per image**.
    Robust to outlier glare/shadow pixels before further scaling.

    Accepts grayscale or multi-channel images; percentiles computed over all channels together
    so behavior matches a single luminance statistic for RGB if you stack channels logically.
    For per-channel percentile clip, loop channels or split into separate calls.
    """
    if low_pct >= high_pct:
        raise ValueError("low_pct must be < high_pct")
    work = img.astype(np.float32, copy=False).ravel()
    lo = float(np.percentile(work, low_pct))
    hi = float(np.percentile(work, high_pct))
    clipped = np.clip(img.astype(np.float32, copy=True), lo, hi)
    return clipped


def robust_intensity_normalize(
    img: np.ndarray,
    low_pct: float = 2.0,
    high_pct: float = 99.0,
    clip_before: bool = True,
    out_dtype=np.float32,
) -> np.ndarray:
    """
    Percentile clip then linear map to ~[0, 1]. Outlier-safe alternative to naive min-max.
    """
    x = clip_intensity_percentiles(img, low_pct, high_pct) if clip_before else img.astype(np.float32)
    lo = float(x.min())
    hi = float(x.max())
    if hi <= lo:
        return np.zeros_like(x, dtype=out_dtype)
    return ((x - lo) / (hi - lo)).astype(out_dtype)


def filter_extreme_dimensions(
    hw: Tuple[int, int],
    max_side: Optional[int] = None,
    min_side: Optional[int] = None,
) -> bool:
    """
    Return True if (height, width) is acceptable, False if it looks like a bad/outlier file
    (e.g. corrupt header, absurd resolution). Tune thresholds for your dataset.
    """
    h, w = hw
    if h <= 0 or w <= 0:
        return False
    if min_side is not None and min(h, w) < min_side:
        return False
    if max_side is not None and max(h, w) > max_side:
        return False
    return True


def preprocess_image(
    bgr: np.ndarray,
    target_hw: Tuple[int, int],
    grayscale: bool = True,
    resize_mode: ResizeMode = "letterbox",
    pad_value: Union[int, float] = 0,
    outlier_percentiles: Optional[Tuple[float, float]] = (2.0, 99.0),
    normalize_01: bool = True,
) -> np.ndarray:
    """
    Full pipeline on a loaded BGR image (as returned by cv2.imread).

    Args:
        bgr: HxWx3 uint8 typically.
        target_hw: (height, width) e.g. (224, 224).
        grayscale: Convert to grayscale before resize/normalize.
        resize_mode: "letterbox" keeps aspect ratio; "stretch" warps to exact size.
        pad_value: border value for letterbox (ignored for stretch).
        outlier_percentiles: (low, high) per-image percentile clip; None to disable.
        normalize_01: Map clipped range to floats in [0, 1].

    Returns:
        Grayscale: HxW float32 in [0,1] if normalize_01 else clipped float/int.
        Color (grayscale=False): HxWx3 same rules.
    """
    x = np.asarray(bgr)
    target_h, target_w = target_hw

    if grayscale:
        x = bgr_to_grayscale(x)

    # Outlier handling on intensities (per image)
    if outlier_percentiles is not None:
        lp, hp = outlier_percentiles
        x = clip_intensity_percentiles(x, lp, hp)

    if resize_mode == "letterbox":
        x = letterbox_resize(x, target_h, target_w, pad_value=pad_value)
    elif resize_mode == "stretch":
        x = stretch_resize(x, target_h, target_w)
    else:
        raise ValueError(resize_mode)

    if normalize_01:
        x = robust_intensity_normalize(
            x,
            low_pct=0,
            high_pct=100,
            clip_before=False,  # already clipped if requested
            out_dtype=np.float32,
        )

    return x


def preprocess_image_path(
    path: Union[str, Path],
    target_hw: Tuple[int, int],
    grayscale: bool = True,
    resize_mode: ResizeMode = "letterbox",
    **kwargs,
) -> np.ndarray:
    """Load (RGB-aware: grayscale + RGBA) and preprocess — see :func:`imread_preprocessable_bgr`."""
    bgr = imread_preprocessable_bgr(path)
    return preprocess_image(bgr, target_hw, grayscale=grayscale, resize_mode=resize_mode, **kwargs)


def iter_exam_cheating_image_paths(
    dataset_root: Path,
    exts: Optional[Iterable[str]] = None,
) -> Iterable[Path]:
    """
    All images under ``train/<class>/`` plus ``test/images/`` (EDA / Kaggle layout).

    Paths are yielded sorted for reproducible runs.
    """
    chosen = frozenset(exts) if exts is not None else DEFAULT_IMAGE_EXTS
    train = dataset_root / "train"
    if train.is_dir():
        for cls_dir in sorted(train.iterdir()):
            if not cls_dir.is_dir():
                continue
            for f in sorted(cls_dir.iterdir()):
                if f.is_file() and f.suffix.lower() in chosen:
                    yield f
    test_imgs = dataset_root / "test" / "images"
    if test_imgs.is_dir():
        for f in sorted(test_imgs.iterdir()):
            if f.is_file() and f.suffix.lower() in chosen:
                yield f


def float_grayscale_01_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Best-effort uint8 grayscale for saving (PNG/JPEG expects 0–255)."""
    if arr.dtype != np.float32 and arr.dtype != np.float64:
        arr = arr.astype(np.float32)
    return np.clip(np.round(arr * 255.0), 0, 255).astype(np.uint8)


def batch_preprocess_exam_cheating(
    dataset_root: Path,
    output_root: Path,
    *,
    skip_existing: bool = True,
    exts: Optional[Iterable[str]] = None,
    progress_every: int = 50,
    output_suffix: str = ".png",
) -> Tuple[int, int, List[str]]:
    """
    Preprocess every image from :func:`iter_exam_cheating_image_paths` and write outputs
    under ``output_root`` **mirroring relative paths** (e.g. ``train/normal act/a.jpg`` →
    ``train/normal act/a.png``).

    Saves **single-channel uint8 grayscale** PNGs ([0–255]), matching the grayscale float
    pipeline from :func:`preprocess_exam_cheating_path`.

    Returns:
        ``(written_count, skipped_count, error_messages)`` — errors cap at first 25 messages.
    """
    root = dataset_root.resolve()
    out_base = output_root.resolve()
    suff = output_suffix.lower() if output_suffix.startswith(".") else f".{output_suffix.lower()}"
    written = 0
    skipped = 0
    errors: List[str] = []
    paths = list(iter_exam_cheating_image_paths(root, exts))
    total = len(paths)

    for i, src in enumerate(paths, start=1):
        try:
            rel = src.relative_to(root)
            dest = out_base / rel.with_suffix(suff)
            if skip_existing and dest.is_file():
                skipped += 1
                if progress_every > 0 and i % progress_every == 0:
                    print(f"[{i}/{total}] skipped (exists) … {rel}")
                continue
            proc = preprocess_exam_cheating_path(src)
            if proc.ndim != 2:
                raise RuntimeError(f"Expected HxW grayscale, got shape {proc.shape}")
            um = float_grayscale_01_to_uint8(proc)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(dest), um):
                raise RuntimeError("cv2.imwrite returned False")
            written += 1
            if progress_every > 0 and written % progress_every == 0:
                print(f"[{i}/{total}] wrote {written} … {dest.relative_to(out_base)}")
        except Exception as ex:  # noqa: BLE001 — batch job: capture and continue
            msg = f"{src.relative_to(root) if src.is_relative_to(root) else src}: {ex}"
            errors.append(msg)
            if len(errors) <= 3:
                print(f"WARN: {msg}")
            elif len(errors) == 4:
                print("WARN: (suppressing further per-file WARN lines; see returned list)")

    if progress_every >= 0:
        print(f"Done. written={written} skipped={skipped} failed={len(errors)} total_files={total}")
    return written, skipped, errors[:25]


__all__ = [
    "ExamCheatingEDA",
    "EXAM_CHEATING_EDA",
    "bgr_to_grayscale",
    "bgra_to_bgr_composite_white",
    "exam_cheating_dimensions_plausible",
    "exam_cheating_preprocess_defaults",
    "imread_preprocessable_bgr",
    "letterbox_resize",
    "stretch_resize",
    "clip_intensity_percentiles",
    "robust_intensity_normalize",
    "preprocess_image",
    "preprocess_image_path",
    "preprocess_exam_cheating_path",
    "filter_extreme_dimensions",
    "DEFAULT_IMAGE_EXTS",
    "iter_exam_cheating_image_paths",
    "batch_preprocess_exam_cheating",
    "float_grayscale_01_to_uint8",
]


def _find_first_demo_image(dataset_root: Path, exts: set[str]) -> Optional[Path]:
    """Prefer train/<class>/*; fallback test/images/* (EDA layout)."""
    train = dataset_root / "train"
    if train.is_dir():
        for cls_dir in sorted(train.iterdir()):
            if not cls_dir.is_dir():
                continue
            for f in sorted(cls_dir.iterdir()):
                if f.is_file() and f.suffix.lower() in exts:
                    return f
    test_imgs = dataset_root / "test" / "images"
    if test_imgs.is_dir():
        for f in sorted(test_imgs.iterdir()):
            if f.is_file() and f.suffix.lower() in exts:
                return f
    return None


def _detect_exam_cheating_dataset_root(
    explicit: Optional[Path],
) -> tuple[Optional[Path], Path, Path]:
    dataset_name = Path(EXAM_CHEATING_EDA.get("dataset_folder", "ExamCheatingDataset"))
    here = Path(__file__).resolve().parent
    cwd = Path.cwd()

    if explicit is not None:
        root = explicit.expanduser().resolve()
        ok = (root / "train").is_dir() or (root / "test" / "images").is_dir()
        return (root if ok else None), here, cwd

    search = [here / dataset_name, cwd / dataset_name, cwd]
    for r in search:
        cand = r if r.name == dataset_name.name or (r / "train").is_dir() else r
        ds = cand if (cand / "train").is_dir() else cand / dataset_name
        if (ds / "train").is_dir() or (ds / "test" / "images").is_dir():
            return ds.resolve(), here, cwd

    return None, here, cwd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OpenCV preprocessing for ExamCheatingDataset (grayscale + letterbox + robust norm).",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Preprocess all images under train/<class>/ and test/images/",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output directory for --batch (mirrors train / test/images layout)",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="ExamCheatingDataset folder (auto-detected next to script or cwd)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-write files even when output PNG already exists",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Log progress every N writes (default 100; 0 to print only summary)",
    )
    parser.add_argument(
        "--no-quick-demo",
        action="store_true",
        help="Skip the single-image demo printed on startup",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Optional image paths — print preprocessing stats without saving",
    )
    cli = parser.parse_args()

    if cli.batch and cli.output is None:
        parser.error("--batch requires -o OUTPUT_DIR")

    ds_root, _here_path, cwd_path = _detect_exam_cheating_dataset_root(cli.dataset_root)

    if cli.batch:
        if ds_root is None:
            parser.error(
                "Could not find ExamCheatingDataset "
                "(use --dataset-root /path/to/ExamCheatingDataset)"
            )
        print("Batch preprocessing")
        print(f"  Dataset: {ds_root}")
        print(f"  Output:  {cli.output.resolve()}")
        _w, _sk, _err = batch_preprocess_exam_cheating(
            ds_root,
            cli.output,
            skip_existing=not cli.force,
            progress_every=cli.progress_every,
        )
        if _err:
            print(f"First errors (up to 25): {_err[:5]}{' ...' if len(_err) > 5 else ''}")
            sys.exit(1)
        sys.exit(0)

    print("EDA-backed defaults:")
    print(" ", exam_cheating_preprocess_defaults())

    _exts = set(DEFAULT_IMAGE_EXTS)
    demo_path = None
    used_root = None
    if ds_root is not None:
        demo_path = _find_first_demo_image(ds_root, _exts)
        used_root = ds_root

    if not cli.no_quick_demo:
        if demo_path is not None:
            print("\nDemo (first image found in dataset layout):")
            print(f"  Dataset root used: {used_root}")
            print(f"  Image: {demo_path}")
            _arr = preprocess_exam_cheating_path(demo_path)
            print(
                f"  → shape={_arr.shape} dtype={_arr.dtype} "
                f"min={float(_arr.min()):.4f} max={float(_arr.max()):.4f}"
            )
        else:
            print("\nCould not find ExamCheatingDataset (train/*/ or test/images/) under:")
            print(f"  script directory: {_here_path}")
            print(f"  cwd:              {cwd_path.resolve()}")
            print("Synthetic in-memory demo:")
            _synth = np.zeros((270, 298, 3), dtype=np.uint8)
            cv2.rectangle(_synth, (50, 40), (220, 200), (180, 90, 40), thickness=-1)
            _d = dict(exam_cheating_preprocess_defaults())
            _d.pop("sanity_dimension_min_side", None)
            _d.pop("sanity_dimension_max_side", None)
            _arr = preprocess_image(_synth, **_d)
            print(
                f"  → shape={_arr.shape} dtype={_arr.dtype} "
                f"min={float(_arr.min()):.4f} max={float(_arr.max()):.4f}"
            )

    for _p in cli.paths:
        _p = _p.expanduser()
        print("\nCLI path:")
        print(f"  {_p}")
        _out = preprocess_exam_cheating_path(_p)
        print(
            f"  → shape={_out.shape} dtype={_out.dtype} "
            f"min={float(_out.min()):.4f} max={float(_out.max()):.4f}"
        )

    if not cli.paths and not cli.no_quick_demo and demo_path is None:
        print("\nBatch all images:")
        print(
            f"  python {Path(sys.argv[0]).name} --batch -o ExamCheatingDataset_preprocessed"
        )
