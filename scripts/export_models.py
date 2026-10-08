#!/usr/bin/env python3
"""
导出「智搜 MediaSeek」所需的两个端侧模型为 Core ML 格式。

产物结构(默认输出到 MediaSeek/Resources/Models/):
  TextEmbedder/
    GemmaText.mlpackage     # EmbeddingGemma 文本嵌入(查询 + 文档/文件名/照片标签)
    tokenizer.json
    meta.json
  ImageEmbedder/
    SiglipImage.mlpackage   # SigLIP 图像编码器
    SiglipText.mlpackage    # SigLIP 文本编码器(与图像同空间,用于自然语言查图)
    tokenizer.json
    meta.json

用法:
  pip install "torch" "transformers>=4.55" "coremltools>=8.0" sentencepiece
  python scripts/export_models.py --outdir MediaSeek/Resources/Models
  python scripts/export_models.py --zip models.zip          # 打成 zip,可在 App 设置里导入

可选环境变量 / 参数:
  --gemma   google/embeddinggemma        # 若 EmbeddingGemma 2 已发布,改为其 HF 仓库名即可,接口一致
  --siglip  google/siglip2-base-b16-256  # 中文查图建议换多语言版本(以 HF 实际仓库名为准)
  --fp16    使用半精度权重(默认开启,体积减半、ANE 加速)
"""
import argparse
import json
import os
import shutil
import zipfile

import numpy as np
import torch
import coremltools as ct
from transformers import AutoModel, AutoTokenizer

GEMMA_DEFAULT = os.environ.get("EMBEDDINGGEMMA_MODEL", "google/embeddinggemma-2")
SIGLIP_DEFAULT = os.environ.get("SIGLIP_MODEL", "google/siglip2-base-patch16-256")

GEMMA_MAX_SEQ = 512   # EmbeddingGemma 上下文内截断
SIGLIP_MAX_SEQ = 64   # SigLIP 文本塔训练序列长度


def save_meta(out_dir: str, meta: dict):
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def copy_tokenizer(tok, out_dir: str):
    fast = getattr(tok, "vocab_file", None)
    path = os.path.join(tok.name_or_path, "tokenizer.json") if hasattr(tok, "name_or_path") else None
    if path and os.path.exists(path):
        shutil.copy(path, os.path.join(out_dir, "tokenizer.json"))
        return
    # 从缓存目录兜底查找
    cached = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    raise RuntimeError(
        "找不到 tokenizer.json(需要 fast tokenizer)。请确认模型仓库包含 tokenizer.json。"
        if not fast else "tokenizer.json 缺失,请检查模型文件。"
    )


def export_gemma(out_root: str, model_id: str, fp16: bool):
    print(f"== 导出文本模型 {model_id} ==")
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, dtype=torch.float32).eval()
    # v2 配置结构有变,pad_token_id / hidden_size 逐级回退
    cfg = model.config
    pad_id = getattr(cfg, "pad_token_id", None)
    if pad_id is None:
        tc = getattr(cfg, "text_config", None)
        pad_id = getattr(tc, "pad_token_id", None) if tc is not None else None
    if pad_id is None:
        pad_id = tok.convert_tokens_to_ids("<pad>")
    if pad_id is None or pad_id < 0:
        pad_id = 0
    hidden = getattr(cfg, "hidden_size", None)
    if hidden is None:
        tc = getattr(cfg, "text_config", None)
        hidden = getattr(tc, "hidden_size", 768) if tc is not None else 768

    class GemmaEmbedding(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.m = model
            self.register_buffer("pad_id", torch.tensor(pad_id))

        def forward(self, input_ids):
            hs = self.m(input_ids=input_ids).last_hidden_state          # [B,L,H]
            mask = (input_ids != self.pad_id).unsqueeze(-1).to(hs.dtype)
            emb = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)  # mean pooling
            return torch.nn.functional.normalize(emb, dim=-1)

    wrapped = GemmaEmbedding().eval()
    ex = torch.ones((1, 16), dtype=torch.int32)
    with torch.no_grad():
        traced = torch.jit.trace(wrapped, ex)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input_ids", shape=(1, ct.RangeDim(1, GEMMA_MAX_SEQ)), dtype=np.int32)],
        outputs=[ct.TensorType(name="embedding")],
        compute_precision=ct.precision.FLOAT16 if fp16 else ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.iOS17,
        convert_to="mlprogram",
    )

    out_dir = os.path.join(out_root, "TextEmbedder")
    os.makedirs(out_dir, exist_ok=True)
    mlmodel.save(os.path.join(out_dir, "GemmaText.mlpackage"))
    copy_tokenizer(tok, out_dir)
    save_meta(out_dir, {"type": "gemma", "space": "gemma", "dim": int(hidden)})
    print(f"  -> {out_dir}")


def export_siglip(out_root: str, model_id: str, fp16: bool):
    print(f"== 导出图文模型 {model_id} ==")
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, dtype=torch.float32).eval()
    cfg = model.config.vision_config
    size = int(getattr(cfg, "image_size", 256))
    mean = getattr(cfg, "image_mean", None) or 0.5
    std = getattr(cfg, "image_std", None) or 0.5
    dim = int(model.config.text_config.hidden_size)

    class SiglipImage(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.m = model

        def forward(self, pixel_values):
            f = self.m.get_image_features(pixel_values=pixel_values)
            return torch.nn.functional.normalize(f, dim=-1)

    class SiglipText(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.m = model

        def forward(self, input_ids):
            f = self.m.get_text_features(input_ids=input_ids)
            return torch.nn.functional.normalize(f, dim=-1)

    img_ex = torch.zeros((1, 3, size, size), dtype=torch.float32)
    with torch.no_grad():
        traced_img = torch.jit.trace(SiglipImage().eval(), img_ex)
        traced_txt = torch.jit.trace(SiglipText().eval(), torch.ones((1, 16), dtype=torch.int32))

    common = dict(
        outputs=[ct.TensorType(name="embedding")],
        compute_precision=ct.precision.FLOAT16 if fp16 else ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.iOS17,
        convert_to="mlprogram",
    )
    img_ml = ct.convert(
        traced_img,
        inputs=[ct.TensorType(name="pixel_values", shape=(1, 3, size, size), dtype=np.float32)],
        **common,
    )
    txt_ml = ct.convert(
        traced_txt,
        inputs=[ct.TensorType(name="input_ids", shape=(1, ct.RangeDim(1, SIGLIP_MAX_SEQ)), dtype=np.int32)],
        **common,
    )

    out_dir = os.path.join(out_root, "ImageEmbedder")
    os.makedirs(out_dir, exist_ok=True)
    img_ml.save(os.path.join(out_dir, "SiglipImage.mlpackage"))
    txt_ml.save(os.path.join(out_dir, "SiglipText.mlpackage"))
    copy_tokenizer(tok, out_dir)
    save_meta(out_dir, {
        "type": "clip", "space": "clip", "dim": dim,
        "image_size": size, "image_mean": float(mean), "image_std": float(std),
    })
    print(f"  -> {out_dir}")


def make_zip(out_root: str, zip_path: str):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(out_root):
            for name in files:
                p = os.path.join(root, name)
                z.write(p, os.path.relpath(p, out_root))
    print(f"已打包 -> {zip_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="MediaSeek/Resources/Models")
    ap.add_argument("--gemma", default=GEMMA_DEFAULT)
    ap.add_argument("--siglip", default=SIGLIP_DEFAULT)
    ap.add_argument("--skip-gemma", action="store_true")
    ap.add_argument("--skip-siglip", action="store_true")
    ap.add_argument("--fp32", action="store_true", help="不量化,使用 float32(默认 fp16)")
    ap.add_argument("--zip", default=None, help="同时打包为 zip(供 App 内导入)")
    args = ap.parse_args()

    fp16 = not args.fp32
    os.makedirs(args.outdir, exist_ok=True)
    if not args.skip_gemma:
        export_gemma(args.outdir, args.gemma, fp16)
    if not args.skip_siglip:
        export_siglip(args.outdir, args.siglip, fp16)
    if args.zip:
        make_zip(args.outdir, args.zip)
    print("完成。接下来: xcodegen generate && open MediaSeek.xcodeproj")


if __name__ == "__main__":
    main()
