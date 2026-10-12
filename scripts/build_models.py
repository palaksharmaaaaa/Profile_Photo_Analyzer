"""
Build the locally-derived models used by the analyzer.

    pip install -r requirements-build.txt
    python scripts/build_models.py                      # everything (about 1 GB of pinned downloads)
    python scripts/build_models.py --only mivolo scrfd  # only some stages (manifest entries are merged)
    python scripts/build_models.py --cleanup            # also delete the scratch download folder afterwards

Stages and outputs (in ``models/``):
  siglip   siglip2_base_vision_int8.onnx + label_embeddings.npz   SigLIP2-B/16 vision tower (INT8) and the prompt
           embeddings produced by its text tower (Apache-2.0, onnx-community export)
  mivolo   mivolo_v2_age_gender.onnx    MiVOLO v2 (face + body transformer). The community export only returns age;
           this stage exposes the network's full head so gender can be read too (graph edit, no retraining)
  scrfd    scrfd_10g.onnx               SCRFD-10G face detector extracted from InsightFace's buffalo_l pack

Every upstream download is verified against a pinned SHA-256 before it is used, and models/manifest.json records the
SHA-256 of every built file. NOTE ON LICENCES: see NOTICE.md - the SCRFD weights are published for non-commercial
research, and the MiVOLO v2 training data is not fully disclosed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def _load_label_prompts():
    """Load the prompt taxonomy by file path so the build needs none of the web-service dependencies."""
    path = ROOT / "app" / "services" / "label_prompts.py"
    spec = importlib.util.spec_from_file_location("label_prompts", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["label_prompts"] = module  # dataclasses resolve their module through sys.modules
    spec.loader.exec_module(module)
    return module


_prompts = _load_label_prompts()
FACE_CLASSES, IMAGE_CLASSES, prompt_templates = _prompts.FACE_CLASSES, _prompts.IMAGE_CLASSES, _prompts.prompt_templates

SIGLIP = "https://huggingface.co/onnx-community/siglip2-base-patch16-224-ONNX/resolve/main"
MIVOLO = "https://huggingface.co/Sam-Apostel/mivolo-v2-age-onnx/resolve/main"
BUFFALO = "https://github.com/deepinsight/insightface/releases/download/v0.7"

# local name -> (url, sha256)
SOURCES = {
    "sig_vision_int8.onnx": (
        f"{SIGLIP}/onnx/vision_model_int8.onnx",
        "0dd31785a2713f1113ef2272472165c69d580473dae38d7b47568ac587795e70",
    ),
    "sig_text_int8.onnx": (
        f"{SIGLIP}/onnx/text_model_int8.onnx",
        "3a0603d3a00c05a80a6ded4743c16aaac7b1e62cdcc7e362e7ce418659b96400",
    ),
    "tokenizer.json": (
        f"{SIGLIP}/tokenizer.json",
        "cb9140fae3ac5122c972d37adf83e1248471a38147ad76f8215c8872c6fd8322",
    ),
    "mivolo_v2_age.onnx": (
        f"{MIVOLO}/mivolo_v2_age.onnx",
        "2db5e05be33b3f120518a86f29a4a65eb0a01c2967219b8787a7dead9b796951",
    ),
    "buffalo_l.zip": (
        f"{BUFFALO}/buffalo_l.zip",
        "80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f",
    ),
}
STAGE_SOURCES = {
    "siglip": ["sig_vision_int8.onnx", "sig_text_int8.onnx", "tokenizer.json"],
    "mivolo": ["mivolo_v2_age.onnx"],
    "scrfd": ["buffalo_l.zip"],
}

OUT_VISION = "siglip2_base_vision_int8.onnx"
OUT_MIVOLO = "mivolo_v2_age_gender.onnx"
OUT_SCRFD = "scrfd_10g.onnx"
OUT_BANK = "label_embeddings.npz"
OUT_MANIFEST = "manifest.json"
SCRFD_MEMBER = "det_10g.onnx"
SCRFD_SHA256 = "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91"
MIVOLO_HEAD_TENSOR = "add_45"  # [1, 3]: gender logits (male, female) then the raw age value, before slicing
TEXT_TOKENS = 64


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, expected: str) -> None:
    if dest.is_file() and sha256_of(dest) == expected:
        print(f"  ok (cached) {dest.name}")
        return
    print(f"  downloading {dest.name} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "ProfilePhotoAnalyzer-build/1.0"})
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as out:  # noqa: S310
        shutil.copyfileobj(resp, out, 1024 * 1024)
    actual = sha256_of(tmp)
    if actual != expected:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"SHA-256 mismatch for {dest.name}: expected {expected}, got {actual}")
    tmp.replace(dest)


def build_mivolo(src: Path, dst: Path) -> dict[str, float]:
    """Expose MiVOLO's full 3-value head (gender logits + raw age) next to the original age output."""
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto, helper

    print("  exposing the gender head of the MiVOLO v2 graph ...")
    model = onnx.load(str(src))
    if not any(MIVOLO_HEAD_TENSOR in node.output for node in model.graph.node):
        raise SystemExit(f"Expected tensor {MIVOLO_HEAD_TENSOR!r} not found: the upstream export has changed.")
    model.graph.output.append(helper.make_tensor_value_info("head", TensorProto.FLOAT, [1, 3]))
    model.graph.node.append(helper.make_node("Identity", [MIVOLO_HEAD_TENSOR], ["head"]))
    onnx.save(model, str(dst))

    session = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(3):
        x = rng.normal(0.0, 1.0, size=(1, 6, 384, 384)).astype(np.float32)
        age, head = session.run(["age", "head"], {"faces_bodies": x})
        worst = max(worst, abs(float(age[0, 0]) - float(head[0, 2] * 122.0 + 61.0)))
    print(f"  age from original output vs from exposed head: max difference {worst:.2e} years")
    if worst > 1e-3:
        raise SystemExit("The exposed head does not reproduce the original age output; refusing to ship it.")
    return {"max_age_difference_years": worst}


def extract_scrfd(zip_path: Path, dst: Path) -> None:
    print("  extracting the SCRFD-10G detector from the buffalo_l pack ...")
    with zipfile.ZipFile(zip_path) as archive:
        data = archive.read(SCRFD_MEMBER)
    if hashlib.sha256(data).hexdigest() != SCRFD_SHA256:
        raise SystemExit(f"{SCRFD_MEMBER} inside buffalo_l.zip does not match the pinned SHA-256.")
    dst.write_bytes(data)


def embed_prompts(text_model: Path, tokenizer_path: Path, prompts: list[str]) -> np.ndarray:
    import onnxruntime as ort
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tokenizer_path))
    tok.enable_padding(length=TEXT_TOKENS, pad_id=0, pad_token="<pad>")  # noqa: S106
    tok.enable_truncation(max_length=TEXT_TOKENS)
    session = ort.InferenceSession(str(text_model), providers=["CPUExecutionProvider"])
    ids = np.array([e.ids for e in tok.encode_batch([p.lower() for p in prompts])], dtype=np.int64)
    pooled = session.run(["pooler_output"], {"input_ids": ids})[0]
    return pooled / np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)


def build_bank(text_model: Path, tokenizer_path: Path, out: Path) -> None:
    print("  embedding prompts with the SigLIP2 text tower ...")
    arrays: dict[str, np.ndarray] = {}
    for taxonomy, classes in (("face", FACE_CLASSES), ("image", IMAGE_CLASSES)):
        flat = [p for c in classes for p in c.prompts]
        emb = embed_prompts(text_model, tokenizer_path, flat)
        vectors, cursor = [], 0
        for c in classes:
            mean = emb[cursor : cursor + len(c.prompts)].mean(axis=0)
            vectors.append(mean / np.linalg.norm(mean))
            cursor += len(c.prompts)
        arrays[f"{taxonomy}_names"] = np.array([c.name for c in classes])
        arrays[f"{taxonomy}_vectors"] = np.stack(vectors).astype(np.float32)
    meta = {"prompts": prompt_templates(), "tokens": TEXT_TOKENS, "text_model": "siglip2-base-patch16-224 (int8)"}
    np.savez(out, meta=np.array(json.dumps(meta)), **arrays)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work-dir", type=Path, default=ROOT / ".model_build" / "dl")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--only", nargs="+", choices=sorted(STAGE_SOURCES), help="build only these stages")
    parser.add_argument("--cleanup", action="store_true", help="delete the scratch download folder when done")
    args = parser.parse_args()

    stages = list(args.only or STAGE_SOURCES)
    work, models = args.work_dir, args.models_dir
    work.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)

    manifest_path = models / OUT_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    manifest.update({"schema": 2})
    manifest.setdefault("files", {})
    manifest.setdefault("upstream_sources", {})
    manifest.setdefault("validation", {})

    print("fetching pinned upstream files")
    for stage in stages:
        for name in STAGE_SOURCES[stage]:
            url, expected = SOURCES[name]
            download(url, work / name, expected)
            manifest["upstream_sources"][name] = expected

    built: list[str] = []
    if "siglip" in stages:
        print("SigLIP2 vision tower + label embeddings")
        shutil.copyfile(work / "sig_vision_int8.onnx", models / OUT_VISION)
        build_bank(work / "sig_text_int8.onnx", work / "tokenizer.json", models / OUT_BANK)
        built += [OUT_VISION, OUT_BANK]
        manifest["label_bank"] = {"file": OUT_BANK, "sha256": sha256_of(models / OUT_BANK)}
    if "mivolo" in stages:
        print("MiVOLO v2 age + gender")
        manifest["validation"]["mivolo_exposed_head"] = build_mivolo(work / "mivolo_v2_age.onnx", models / OUT_MIVOLO)
        built.append(OUT_MIVOLO)
    if "scrfd" in stages:
        print("SCRFD-10G face detector")
        extract_scrfd(work / "buffalo_l.zip", models / OUT_SCRFD)
        built.append(OUT_SCRFD)

    import onnxruntime

    manifest["onnxruntime_version"] = onnxruntime.__version__
    for name in built:
        if name.endswith(".onnx"):
            manifest["files"][name] = {"sha256": sha256_of(models / name), "size": (models / name).stat().st_size}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("built:", ", ".join([*built, OUT_MANIFEST]))

    if args.cleanup:
        shutil.rmtree(work, ignore_errors=True)
        print("scratch folder removed")


if __name__ == "__main__":
    main()
