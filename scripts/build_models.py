"""
Build the locally-derived models used by the analyzer.

    pip install -r requirements-build.txt
    python scripts/build_models.py            # downloads ~760 MB of upstream files, builds, verifies
    python scripts/build_models.py --cleanup  # also delete the scratch download folder afterwards

Outputs (in ``models/``):
  * siglip2_base_vision_int8.onnx   SigLIP2-B/16 vision tower, INT8 (from onnx-community, Apache-2.0)
  * age_gender_vit_int8.onnx        abhilash88 age/gender ViT, dynamically quantised to INT8
  * label_embeddings.npz            prompt embeddings produced by the SigLIP2 text tower
  * manifest.json                   SHA-256 of every output and of every upstream input

Every upstream download is verified against a pinned SHA-256 before it is used.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import urllib.request
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
AGEGENDER = "https://huggingface.co/onnx-community/age-gender-prediction-ONNX/resolve/main"

# (local name, url, sha256)
SOURCES = {
    "siglip_vision": ("sig_vision_int8.onnx", f"{SIGLIP}/onnx/vision_model_int8.onnx"),
    "siglip_text": ("sig_text_int8.onnx", f"{SIGLIP}/onnx/text_model_int8.onnx"),
    "siglip_tokenizer": ("tokenizer.json", f"{SIGLIP}/tokenizer.json"),
    "age_gender_fp32": ("agegender_fp32.onnx", f"{AGEGENDER}/onnx/model.onnx"),
}
SOURCE_SHA256 = {
    "sig_vision_int8.onnx": "0dd31785a2713f1113ef2272472165c69d580473dae38d7b47568ac587795e70",
    "sig_text_int8.onnx": "3a0603d3a00c05a80a6ded4743c16aaac7b1e62cdcc7e362e7ce418659b96400",
    "tokenizer.json": "cb9140fae3ac5122c972d37adf83e1248471a38147ad76f8215c8872c6fd8322",
    "agegender_fp32.onnx": "0c35c868ea8ffba6d5fe727c1a6b82d9da600e690342d58ca268270560e28304",
}

OUT_VISION = "siglip2_base_vision_int8.onnx"
OUT_AGE_GENDER = "age_gender_vit_int8.onnx"
OUT_BANK = "label_embeddings.npz"
OUT_MANIFEST = "manifest.json"
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


def quantize_age_gender(src: Path, dst: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    print("  quantising age/gender ViT to INT8 ...")
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"])


def validate_quantisation(fp32: Path, int8: Path) -> dict[str, float]:
    """Refuse to ship an INT8 model whose outputs drift from the fp32 reference."""
    import onnxruntime as ort

    ref = ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"])
    quant = ort.InferenceSession(str(int8), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    batch = rng.normal(0.0, 1.0, size=(8, 3, 224, 224)).astype(np.float32)
    a = ref.run(None, {"pixel_values": batch})[0]
    b = quant.run(None, {"pixel_values": batch})[0]
    age_drift = float(np.mean(np.abs(a[:, 0] - b[:, 0])))
    gender_drift = float(np.mean(np.abs(a[:, 1] - b[:, 1])))
    print(f"  INT8 vs fp32 drift: age {age_drift:.2f} y, gender {gender_drift:.3f}")
    if age_drift > 3.0 or gender_drift > 0.15:
        raise SystemExit("INT8 model drifts too far from the fp32 reference; refusing to ship it.")
    return {"mean_abs_age_drift_years": round(age_drift, 3), "mean_abs_gender_drift": round(gender_drift, 4)}


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
    parser.add_argument("--cleanup", action="store_true", help="delete the scratch download folder when done")
    args = parser.parse_args()

    work, models = args.work_dir, args.models_dir
    work.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)

    print("1/4 fetching pinned upstream files")
    for local, url in SOURCES.values():
        download(url, work / local, SOURCE_SHA256[local])

    print("2/4 SigLIP2 vision tower")
    shutil.copyfile(work / "sig_vision_int8.onnx", models / OUT_VISION)

    print("3/4 age/gender ViT")
    quantize_age_gender(work / "agegender_fp32.onnx", models / OUT_AGE_GENDER)
    drift = validate_quantisation(work / "agegender_fp32.onnx", models / OUT_AGE_GENDER)

    print("4/4 label embeddings")
    build_bank(work / "sig_text_int8.onnx", work / "tokenizer.json", models / OUT_BANK)

    import onnxruntime

    manifest = {
        "schema": 1,
        "onnxruntime_version": onnxruntime.__version__,
        "files": {
            name: {"sha256": sha256_of(models / name), "size": (models / name).stat().st_size}
            for name in (OUT_VISION, OUT_AGE_GENDER)
        },
        "label_bank": {"file": OUT_BANK, "sha256": sha256_of(models / OUT_BANK)},
        "upstream_sources": SOURCE_SHA256,
        "quantisation_validation": drift,
    }
    (models / OUT_MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("built:", ", ".join([OUT_VISION, OUT_AGE_GENDER, OUT_BANK, OUT_MANIFEST]))

    if args.cleanup:
        shutil.rmtree(work, ignore_errors=True)
        print("scratch folder removed")


if __name__ == "__main__":
    main()
