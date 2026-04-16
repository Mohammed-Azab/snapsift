# snapsift

Sift your Apple Photos library into three clean folders using on-device face
recognition:

1. **`no_people/`** — photos with no faces
2. **`solo_without_me/`** — photos with exactly one face that isn't yours
3. **`with_me_or_group/`** — photos of you, or group shots

Anything that can't be decoded lands in `unreadable/` so nothing is lost.

Runs locally on your Mac. No photos leave your machine.

## Why it copies instead of moves

An Apple Photos library (`*.photoslibrary`) is a managed package — moving
original files out of it corrupts the database. `snapsift` copies originals
into your sorted folders and leaves the library untouched. Same end result
for you, zero risk of breaking Photos.

## Setup

```bash
# Apple Silicon needs cmake for dlib (a face_recognition dependency)
brew install cmake

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Grant your terminal **Full Disk Access**:
System Settings → Privacy & Security → Full Disk Access. Without it,
`osxphotos` can't read your library.

## Usage

```bash
# 1. Put 3–5 clear, forward-facing photos of just your face in ./me/
# 2. Run:
python sort_photos.py --reference ./me/*.jpg --output ~/Pictures/SortedPhotos
```

Test first on a subset without writing anything:

```bash
python sort_photos.py --reference ./me/*.jpg --output ./out --limit 100 --dry-run
```

## Tuning

| flag | default | what it does |
|---|---|---|
| `--tolerance` | `0.55` | Lower = stricter "is this me?" match. Raise to `0.6` if you're being missed in photos; lower to `0.5` if strangers are matching as you. |
| `--model` | `hog` | `hog` is CPU-fast; `cnn` is more accurate but slow on CPU. |
| `--limit` | *(none)* | Process only first N photos — useful for trial runs. |
| `--dry-run` | off | Print what would happen; no files written. |

## How it works

1. Loads your reference photos and computes a 128-d face embedding for each.
2. Walks the Photos library via `osxphotos` to get original file paths.
3. For each photo: downscales, detects faces, computes embeddings, compares
   to your reference set via Euclidean distance.
4. Routes to a bucket based on face count and whether any face matched you.

## License

MIT
