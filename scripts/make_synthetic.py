"""Build a SYNTHETIC sanity-check video from Tears of Steel (CC-BY 3.0, Blender Foundation).

Five runs of consecutive shots are taken from the film and concatenated in a new
order. Inside each run the editing is the film's own (including a real A/B
dialogue), and the joins between runs are by construction scene changes. This
gives a video whose scene boundaries are known without anyone labelling them.

It is NOT a substitute for human-labelled evaluation data. It only checks that
the pipeline, the baselines and the metrics behave sensibly end to end. It is
written to data/synthetic/, never to ground_truth/.

    python scripts/make_synthetic.py            # one video  -> data/synthetic/synthetic_tos.mp4
    python scripts/make_synthetic.py --set 4    # + 4 re-ordered variants with labels in data/synthetic/labels/
                                                #   (used by config/synthetic_check.toml to exercise
                                                #    evaluate/tune; numbers are NOT a real benchmark)
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sceneseen.media import ffmpeg_exe  # noqa: E402

SRC = ROOT / "data/videos/tears_of_steel_720p.mov"
OUT_DIR = ROOT / "data/synthetic"

# (start, end) in seconds of shot runs in the source film; the times are shot-cut
# times found by the shot detector, so every join is on a real cut.
SEGMENTS = [
    (383.125, 415.000, "bridge dialogue, alternating close-ups (A/B/A/B)"),
    (152.875, 193.083, "church interior, warm light, group conversation"),
    (82.125, 152.875, "sniper position, blue night exterior"),
    (478.958, 509.958, "robot battle inside the church"),
    (310.583, 330.292, "bridge memory, daylight"),
]


def build(segments, out: Path) -> list[float]:
    parts, labels = [], []
    for i, (a, b, _) in enumerate(segments):
        parts.append(f"[0:v]trim=start={a}:end={b},setpts=PTS-STARTPTS,scale=854:-2[v{i}];"
                     f"[0:a]atrim=start={a}:end={b},asetpts=PTS-STARTPTS[a{i}];")
        labels.append(f"[v{i}][a{i}]")
    fc = "".join(parts) + "".join(labels) + f"concat=n={len(segments)}:v=1:a=1[v][a]"
    cmd = [ffmpeg_exe(), "-v", "error", "-y", "-i", str(SRC), "-filter_complex", fc, "-map", "[v]", "-map", "[a]",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "aac", "-r", "24", str(out)]
    subprocess.run(cmd, check=True)
    t, bounds = 0.0, []
    for a, b, _ in segments[:-1]:
        t += b - a
        bounds.append(round(t, 3))
    return bounds


def label_for(out: Path, segments, bounds) -> dict:
    return {
        "video": out.name,
        "boundaries": bounds,
        "annotator": "constructed (synthetic concatenation, not a human label)",
        "segments": [{"source_start": a, "source_end": b, "description": d} for a, b, d in segments],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", type=int, default=0, help="also build N re-ordered variants for the machinery check")
    args = ap.parse_args()
    if not SRC.exists():
        sys.exit(f"missing {SRC}; download from https://download.blender.org/demo/movies/ToS/")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "synthetic_tos.mp4"
    bounds = build(SEGMENTS, out)
    (OUT_DIR / "synthetic_tos.json").write_text(json.dumps(label_for(out, SEGMENTS, bounds), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}, boundaries={bounds}")
    if args.set:
        labels_dir = OUT_DIR / "labels"
        labels_dir.mkdir(exist_ok=True)
        rng = random.Random(7)
        for k in range(1, args.set + 1):
            segs = SEGMENTS[:]
            rng.shuffle(segs)
            segs = segs[: rng.randint(3, len(segs))]
            out_k = OUT_DIR / f"synthetic_tos_v{k}.mp4"
            b = build(segs, out_k)
            (labels_dir / f"{out_k.stem}.json").write_text(json.dumps(label_for(out_k, segs, b), indent=2) + "\n", encoding="utf-8")
            print(f"wrote {out_k}, boundaries={b}")


if __name__ == "__main__":
    main()
