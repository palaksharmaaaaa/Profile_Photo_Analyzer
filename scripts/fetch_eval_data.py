"""
Download labelled evaluation images into ``.eval_data/`` (git-ignored). Nothing is redistributed by this repository.

    pip install -r requirements-eval.txt
    python scripts/fetch_eval_data.py [--faces 1200] [--per-set 120] [--seed 7] [--out .eval_data]

Sources (Hugging Face datasets; check each dataset's own licence before reusing the images for anything other than
evaluating this service):

  fairface   HuggingFaceM4/FairFace, config 1.25, validation split (one 237 MB parquet file). Real human faces with
             an age bin, a gender and a race group.
  pets       enterprise-explorers/oxford-pets        cats and dogs
  birds      bentrevett/caltech-ucsd-birds-200-2011  birds
  anime      puruchinera/anime-faces-256             anime faces
  cartoon    huggan/cartoon-faces                    cartoon faces
  paintings  huggan/wikiart (genre "portrait")       painted portraits of people
  sketches   a573p/Face-Sketches40                   pencil face sketches

Every set except ``fairface`` holds images that must NOT be reported as a real human face. The sample is random with
a fixed seed, so the same arguments fetch the same images, and a second run continues where an interrupted one
stopped. ``labels.csv`` records, per image, the set, the expectation and (FairFace) the age bin, gender and race.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

API = "https://datasets-server.huggingface.co"
HEADERS = {"User-Agent": "ProfilePhotoAnalyzer-eval/1.0"}
FAIRFACE_PARQUET = "https://huggingface.co/datasets/HuggingFaceM4/FairFace/resolve/refs%2Fconvert%2Fparquet/1.25/validation/0000.parquet"
FAIRFACE_AGES = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "more than 70"]
FAIRFACE_GENDERS = ["Male", "Female"]
FAIRFACE_RACES = ["East Asian", "Indian", "Black", "White", "Middle Eastern", "Latino_Hispanic", "Southeast Asian"]
PAUSE_SECONDS = 0.4  # between requests, to stay under the API's rate limit
WINDOW = 10  # rows fetched together at one random offset, to keep the sample spread over the dataset

# set name -> (dataset, config, split, (column, value) row filter or None)
NON_HUMAN_SETS = {
    "pets": ("enterprise-explorers/oxford-pets", "default", "train", None),
    "birds": ("bentrevett/caltech-ucsd-birds-200-2011", "default", "train", None),
    "anime": ("puruchinera/anime-faces-256", "default", "train", None),
    "cartoon": ("huggan/cartoon-faces", "default", "train", None),
    "paintings": ("huggan/wikiart", "default", "train", ("genre", 6)),  # genre 6 = portrait
    "sketches": ("a573p/Face-Sketches40", "default", "train", None),
}


class RowWriter(Protocol):
    """What the fetch functions need from a ``csv.writer``."""

    def writerow(self, row: Sequence[object], /) -> object: ...


DatasetSpec = tuple[str, str, str, tuple[str, int] | None]
LABELS = ["file", "set", "expectation", "age", "gender", "race"]


RETRYABLE_STATUS = (429, 500, 502, 503, 504)
MAX_RETRIES = 6


def _open(url: str) -> bytes:
    """GET with a patient retry: 429 and 5xx answers are waited out with a growing pause."""
    attempt = 0
    while True:
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=120) as response:
                data: bytes = response.read()
            time.sleep(PAUSE_SECONDS)
            return data
        except urllib.error.HTTPError as exc:
            if exc.code not in RETRYABLE_STATUS or attempt >= MAX_RETRIES:
                raise
        except TimeoutError, urllib.error.URLError:
            if attempt >= MAX_RETRIES:
                raise
        time.sleep(5 * 2**attempt)
        attempt += 1


def _json(url: str) -> dict[str, Any]:
    parsed: dict[str, Any] = json.loads(_open(url))
    return parsed


def _rows(dataset: str, config: str, split: str, offset: int, length: int) -> list[dict[str, Any]]:
    query = {"dataset": dataset, "config": config, "split": split, "offset": offset, "length": length}
    rows: list[dict[str, Any]] = _json(f"{API}/rows?{urllib.parse.urlencode(query)}")["rows"]
    return rows


def _total(dataset: str, config: str) -> int:
    size = _json(f"{API}/size?dataset={urllib.parse.quote(dataset)}&config={config}")
    return int(size["size"]["config"]["num_rows"])


def fetch_fairface(out: Path, count: int, rng: random.Random, writer: RowWriter, known: set[str]) -> None:
    import pyarrow.parquet as pq

    folder = out / "fairface"
    folder.mkdir(parents=True, exist_ok=True)
    parquet = out / "fairface_validation.parquet"
    if not parquet.is_file():
        print("  downloading the FairFace validation parquet (237 MB) ...", flush=True)
        parquet.write_bytes(_open(FAIRFACE_PARQUET))
    table = pq.read_table(parquet)
    total = table.num_rows
    wanted = sorted(rng.sample(range(total), min(count, total)))
    done = 0
    for offset in wanted:
        name = f"fairface/ff_{offset:05d}.jpg"
        if name in known:
            continue
        row = table.slice(offset, 1).to_pylist()[0]
        (folder / Path(name).name).write_bytes(row["image"]["bytes"])
        writer.writerow(
            [name, "fairface", "human", FAIRFACE_AGES[row["age"]], FAIRFACE_GENDERS[row["gender"]],
             FAIRFACE_RACES[row["race"]]]
        )  # fmt: skip
        done += 1
    print(f"  fairface: {done} new images ({len(wanted)} selected)", flush=True)


def _window_rows(name: str, spec: DatasetSpec, offset: int, known: set[str]) -> list[dict[str, Any]]:
    """The rows of one window that match the set's filter and are not yet known; an unavailable window gives none."""
    dataset, config, split, row_filter = spec
    try:
        rows = _rows(dataset, config, split, offset, WINDOW)
    except urllib.error.HTTPError as exc:  # a few windows of a dataset can be unavailable: take the next one
        print(f"  {name}: skipping rows {offset}-{offset + WINDOW} ({exc.code})", flush=True)
        return []
    matching = [row for row in rows if not row_filter or row["row"][row_filter[0]] == row_filter[1]]
    return [row for row in matching if f"{name}/{name}_{row['row_idx']:06d}.jpg" not in known]


def fetch_non_human(
    out: Path, name: str, spec: DatasetSpec, count: int, rng: random.Random, writer: RowWriter, known: set[str]
) -> None:
    dataset, config = spec[0], spec[1]
    total = _total(dataset, config)
    (out / name).mkdir(parents=True, exist_ok=True)
    have = sum(1 for key in known if key.startswith(f"{name}/"))
    windows = list(range(0, max(1, total - WINDOW + 1), WINDOW))
    rng.shuffle(windows)
    for offset in windows:
        if have >= count:
            break
        for row in _window_rows(name, spec, offset, known)[: count - have]:
            file = f"{name}/{name}_{row['row_idx']:06d}.jpg"
            (out / file).write_bytes(_open(row["row"]["image"]["src"]))
            writer.writerow([file, name, "not_human", "", "", ""])
            known.add(file)
            have += 1
    print(f"  {name}: {have} images", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path(".eval_data"))
    parser.add_argument("--faces", type=int, default=1200, help="FairFace faces to take")
    parser.add_argument("--per-set", type=int, default=120, help="images per non-human set")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--only", nargs="+", choices=["fairface", *NON_HUMAN_SETS])
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    labels_path = args.out / "labels.csv"
    known = set()
    if labels_path.is_file():
        with labels_path.open(encoding="utf-8", newline="") as handle:
            known = {row["file"] for row in csv.DictReader(handle)}
    new_file = not labels_path.is_file()

    rng = random.Random(args.seed)
    selected = args.only or ["fairface", *NON_HUMAN_SETS]
    with labels_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if new_file:
            writer.writerow(LABELS)
        for name in selected:
            try:
                if name == "fairface":
                    fetch_fairface(args.out, args.faces, rng, writer, known)
                else:
                    fetch_non_human(args.out, name, NON_HUMAN_SETS[name], args.per_set, rng, writer, known)
            except Exception as exc:  # one unavailable source must not stop the others
                print(f"  {name}: FAILED ({type(exc).__name__}: {exc})", flush=True)
            handle.flush()
    print(f"labels in {labels_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
