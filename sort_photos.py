#!/usr/bin/env python3
"""
sort_photos.py
==============
Scans your Apple Photos library and sorts every photo into one of three buckets
based on face detection + face recognition:

    1. no_people/         — no faces detected
    2. solo_without_me/   — exactly one face, and it isn't you
    3. with_me_or_group/  — you are in the photo, or it's a group shot

Anything that fails to decode goes to `unreadable/` so nothing is lost silently.

WHY WE COPY INSTEAD OF MOVE
---------------------------
An Apple Photos library (`*.photoslibrary`) is a managed package. Moving
original files out of it corrupts the database and you lose the ability to
open those photos in the Photos app. This script therefore COPIES originals
into your output folders. The net result for you is identical: you get tidy
folders organized by content; your library is untouched and still works.

SETUP
-----
    # 1. Apple Silicon: install cmake first (needed by dlib, a face_recognition dep)
    brew install cmake

    # 2. Python deps
    python3 -m venv .venv && source .venv/bin/activate
    pip install --upgrade pip
    pip install face_recognition osxphotos pillow pillow-heif numpy tqdm

    # 3. Give Terminal (or your IDE) "Full Disk Access":
    #    System Settings -> Privacy & Security -> Full Disk Access
    #    Otherwise osxphotos cannot read the Photos library.

USAGE
-----
    python sort_photos.py \
        --reference ./me/*.jpg \
        --output ~/Pictures/SortedPhotos

    # Test on just the first 100 photos without touching anything:
    python sort_photos.py --reference ./me/*.jpg --output ./out --limit 100 --dry-run

TUNING
------
    --tolerance 0.55   Lower = stricter "is this me?" match. 0.6 is the
                       face_recognition default; 0.5–0.55 is more conservative.
    --model hog        CPU-only, fast, slightly less accurate (default).
    --model cnn        GPU-accelerated, more accurate, much slower on CPU.
"""

from __future__ import annotations

import argparse
import glob
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    print("warning: pillow-heif not installed; HEIC photos will be skipped", file=sys.stderr)

import face_recognition
import osxphotos

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **_):  # fallback no-op
        return it


log = logging.getLogger("sort_photos")

# ---------- image loading ----------------------------------------------------

def load_as_rgb_array(path: Path, max_dim: int = 1600) -> np.ndarray | None:
    """Load image as an RGB numpy array, EXIF-rotated and downscaled for speed."""
    try:
        pil = Image.open(path)
        pil = ImageOps.exif_transpose(pil)
        pil = pil.convert("RGB")
        pil.thumbnail((max_dim, max_dim))
        return np.array(pil)
    except Exception as e:
        log.warning("could not read %s: %s", path, e)
        return None


# ---------- reference encoding ----------------------------------------------

def load_reference_encodings(paths: list[Path]) -> np.ndarray:
    encs = []
    for p in paths:
        img = load_as_rgb_array(p, max_dim=1600)
        if img is None:
            continue
        faces = face_recognition.face_encodings(img)
        if not faces:
            log.warning("no face in reference photo: %s", p)
            continue
        if len(faces) > 1:
            log.warning("multiple faces in %s — using the largest / first", p)
        encs.append(faces[0])
    if not encs:
        raise SystemExit(
            "No usable reference photos. Provide 3–5 clear, well-lit, forward-facing "
            "photos of just your face."
        )
    log.info("built reference profile from %d photo(s)", len(encs))
    return np.stack(encs)


# ---------- classification ---------------------------------------------------

def classify(
    image_path: Path,
    ref_encodings: np.ndarray,
    tolerance: float,
    model: str,
) -> str:
    img = load_as_rgb_array(image_path)
    if img is None:
        return "unreadable"

    locations = face_recognition.face_locations(img, model=model)
    if not locations:
        return "no_people"

    encodings = face_recognition.face_encodings(img, known_face_locations=locations)

    me_present = False
    for enc in encodings:
        distances = np.linalg.norm(ref_encodings - enc, axis=1)
        if distances.min() <= tolerance:
            me_present = True
            break

    if len(encodings) == 1 and not me_present:
        return "solo_without_me"
    return "with_me_or_group"


# ---------- library iteration ------------------------------------------------

def iter_library_photos(library_path: str | None):
    db = osxphotos.PhotosDB(dbfile=library_path) if library_path else osxphotos.PhotosDB()
    photos = [p for p in db.photos(images=True, movies=False) if p.path and Path(p.path).exists()]
    log.info("found %d photos with on-disk originals in library", len(photos))
    return photos


# ---------- file operations --------------------------------------------------

def safe_copy(src: Path, dst_dir: Path, dry_run: bool = False) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    i = 1
    while dst.exists():
        dst = dst_dir / f"{src.stem}_{i}{src.suffix}"
        i += 1
    if dry_run:
        log.debug("DRY-RUN copy %s -> %s", src, dst)
    else:
        shutil.copy2(src, dst)
    return dst


# ---------- main -------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--reference", required=True, nargs="+",
                    help="One or more glob patterns for reference photos of you")
    ap.add_argument("--output", required=True, type=Path,
                    help="Output folder; three subfolders will be created inside")
    ap.add_argument("--photos-library", default=None,
                    help="Path to .photoslibrary (default: system library)")
    ap.add_argument("--tolerance", type=float, default=0.55,
                    help="Face match tolerance; lower = stricter (default 0.55)")
    ap.add_argument("--model", choices=["hog", "cnn"], default="hog",
                    help="Face detection model (default: hog)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process the first N photos — useful for testing")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would happen; do not write files")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # expand reference globs
    ref_paths: list[Path] = []
    for pat in args.reference:
        ref_paths.extend(Path(p) for p in glob.glob(str(Path(pat).expanduser())))
    if not ref_paths:
        raise SystemExit(f"No files matched --reference patterns: {args.reference}")

    ref_encodings = load_reference_encodings(ref_paths)

    buckets = {
        "no_people":        args.output / "no_people",
        "solo_without_me":  args.output / "solo_without_me",
        "with_me_or_group": args.output / "with_me_or_group",
        "unreadable":       args.output / "unreadable",
    }
    totals = {k: 0 for k in buckets}

    photos = iter_library_photos(args.photos_library)
    if args.limit:
        photos = photos[: args.limit]

    for photo in tqdm(photos, desc="classifying", unit="img"):
        src = Path(photo.path)
        bucket = classify(src, ref_encodings, args.tolerance, args.model)
        totals[bucket] += 1
        safe_copy(src, buckets[bucket], dry_run=args.dry_run)

    log.info("done. totals: %s", totals)
    print("\nSummary:")
    for k, v in totals.items():
        print(f"  {k:<18} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
