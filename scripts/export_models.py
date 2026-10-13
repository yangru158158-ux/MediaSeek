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


def load_auto(model_id: str):
    """v5 用 dtype,v4 只认 torch_dtype"""
    from transformers import AutoModel
    try:
        return AutoModel.from_pretrained(model_id, dtype=torch.float32).eval()
    except TypeError:
        return AutoModel.from_pretrained(model_id, torch_dtype=torch.float32).eval()


def save_meta(out_dir: str, meta: dict):
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def copy_tokenizer(model_id: str, out_dir: str):
    from huggingface_hub import hf_hub_download
    src = hf_hub_download(model_id, "tokenizer.json")
    dst = os.path.join(out_dir, "tokenizer.json")
    if os.path.exists(dst):
        os.remove(dst)  # HF 缓存文件只读,shutil.copy 会把权限带过来,二次导出覆盖时 PermissionError
    shutil.copy(src, dst)
    # swift-transformers 加载时需要 config.json / tokenizer_config.json
    for name in ("config.json", "tokenizer_config.json"):
        try:
            f = hf_hub_download(model_id, name)
            d = os.path.join(out_dir, name)
            if os.path.exists(d):
                os.remove(d)
            shutil.copy(f, d)
        except Exception as e:
            print(f"  警告: {name} 下载失败({e}),App 端会自动补最小配置")


def export_gemma(out_root: str, model_id: str, fp16: bool):
    print(f"== 导出文本模型 {model_id} ==")
    tok = AutoTokenizer.from_pretrained(model_id)
    model = load_auto(model_id)
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
    copy_tokenizer(model_id, out_dir)
    save_meta(out_dir, {"type": "gemma", "space": "gemma", "dim": int(hidden)})
    print(f"  -> {out_dir}")


def export_siglip(out_root: str, model_id: str, fp16: bool, txt_fp32: bool = False):
    print(f"== 导出图文模型 {model_id} ==")
    tok = AutoTokenizer.from_pretrained(model_id)
    model = load_auto(model_id)
    cfg = model.config.vision_config
    size = int(getattr(cfg, "image_size", 256))
    mean = getattr(cfg, "image_mean", None) or 0.5
    std = getattr(cfg, "image_std", None) or 0.5
    dim = int(model.config.text_config.hidden_size)

    def _features_to_tensor(x):
        # transformers5 的 get_*_features 可能返回 ModelOutput 对象而非张量
        if torch.is_tensor(x):
            return x
        for attr in ("pooler_output", "image_features", "text_features", "last_hidden_state"):
            v = getattr(x, attr, None)
            if v is not None:
                return v
        return x[0]

    class SiglipImage(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.m = model

        def forward(self, pixel_values):
            f = _features_to_tensor(self.m.get_image_features(pixel_values=pixel_values))
            return torch.nn.functional.normalize(f, dim=-1)

    class SiglipText(torch.nn.Module):
        # SigLIP2 文本池化取序列最后一个位置(训练时按 max_length=64 右填充,锚点在 pad 位,
        # pad 查询对真实 token 做注意力=另一个池化头);变长不填充会把锚点落在 EOS 位=没学过的位置。
        # 不走 HF masking_utils(v5 的动态掩码闭包含 coremltools 未注册算子;v4 的 argmax 池化对
        # SigLIP2 是错位),手工构建 4D 加性掩码直通 encoder,语义与官方 _prepare_4d 等价。
        def __init__(self):
            super().__init__()
            self.m = model

        def forward(self, input_ids, attention_mask):
            tm = self.m.text_model
            emb = tm.embeddings(input_ids=input_ids)
            m4 = (1.0 - attention_mask[:, None, None, :].to(emb.dtype)) * -30000.0
            out = tm.encoder(inputs_embeds=emb, attention_mask=m4)
            h = tm.final_layer_norm(out.last_hidden_state)
            pooled = tm.head(h[:, -1, :])
            return torch.nn.functional.normalize(pooled, dim=-1)

    img_ex = torch.zeros((1, 3, size, size), dtype=torch.float32)
    txt_ex = (torch.ones((1, SIGLIP_MAX_SEQ), dtype=torch.int32),
              torch.ones((1, SIGLIP_MAX_SEQ), dtype=torch.int32))
    with torch.no_grad():
        traced_img = torch.jit.trace(SiglipImage().eval(), img_ex)
        traced_txt = torch.jit.trace(SiglipText().eval(), txt_ex)

    common = dict(
        outputs=[ct.TensorType(name="embedding")],
        compute_precision=ct.precision.FLOAT16 if fp16 else ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.iOS17,
        convert_to="mlprogram",
    )
    # 文本塔可选 fp32:v4 加性掩码的 finfo.min(-3.4e38)在 fp16 溢出,实测 CoreML 一致性掉到 0.72;
    # 文本塔每查询只跑一次,fp32 体积+110MB 换数值稳定值得(体检 CI 定案用哪档)
    txt_precision = ct.precision.FLOAT32 if (txt_fp32 or not fp16) else ct.precision.FLOAT16
    img_ml = ct.convert(
        traced_img,
        inputs=[ct.TensorType(name="pixel_values", shape=(1, 3, size, size), dtype=np.float32)],
        **common,
    )
    txt_ml = ct.convert(
        traced_txt,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, SIGLIP_MAX_SEQ), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(1, SIGLIP_MAX_SEQ), dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="embedding")],
        compute_precision=txt_precision,
        minimum_deployment_target=ct.target.iOS17,
        convert_to="mlprogram",
    )

    out_dir = os.path.join(out_root, "ImageEmbedder")
    os.makedirs(out_dir, exist_ok=True)
    img_ml.save(os.path.join(out_dir, "SiglipImage.mlpackage"))
    txt_ml.save(os.path.join(out_dir, "SiglipText.mlpackage"))
    copy_tokenizer(model_id, out_dir)
    save_meta(out_dir, {
        "type": "clip", "space": "clip", "dim": dim,
        "image_size": size, "image_mean": float(mean), "image_std": float(std),
    })
    print(f"  -> {out_dir}")


def export_face(out_root: str, onnx_path: str, fp16: bool):
    """人脸识别模型(insightface MobileFaceNet / ArcFace 系,512 维)。
    检测与关键点用系统 Vision,这里只转识别塔:
    输入 input.1 (N,3,112,112),RGB,(px-127.5)/128;输出未归一化,App 端统一归一化。
    注意:输入/输出名保留 ONNX 原名(如输出 '516'),写进 meta.json 供 App 读取。"""
    import onnx
    print(f"== 导出人脸模型 {onnx_path} ==")
    m = onnx.load(onnx_path)
    in_name = m.graph.input[0].name
    out_name = m.graph.output[0].name
    dim = None
    for d in m.graph.output[0].type.tensor_type.shape.dim:
        if d.dim_value > 1:
            dim = int(d.dim_value)
    print(f"  ONNX 签名: 输入 {in_name} → 输出 {out_name} {dim or '?'} 维")

    mlmodel = ct.convert(
        onnx_path,   # 必须传路径:coremltools 按扩展名识别 ONNX,传 ModelProto 对象会报「无法判断来源框架」
        inputs=[ct.TensorType(name=in_name, shape=(1, 3, 112, 112), dtype=np.float32)],
        compute_precision=ct.precision.FLOAT16 if fp16 else ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.iOS17,
        convert_to="mlprogram",
    )

    # 一致性对拍:同一随机输入,CoreML vs ONNX 原模型(转换错误的保险丝)
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        rng = np.random.RandomState(0)
        x = rng.rand(1, 3, 112, 112).astype(np.float32)
        ref = sess.run([out_name], {in_name: x})[0].flatten()
        pred = np.array(mlmodel.predict({in_name: x})[out_name], dtype=np.float32).flatten()
        cos = float(np.dot(ref, pred) / (np.linalg.norm(ref) * np.linalg.norm(pred) + 1e-9))
        print(f"  CoreML↔ONNX 一致性 cos={cos:.5f}")
        if cos < 0.995:
            raise RuntimeError(f"人脸模型 CoreML 转换一致性不足(cos={cos:.4f})")
    except ImportError:
        print("  (未安装 onnxruntime,跳过一致性对拍)")

    out_dir = os.path.join(out_root, "FaceEmbedder")
    os.makedirs(out_dir, exist_ok=True)
    mlmodel.save(os.path.join(out_dir, "Face.mlpackage"))
    save_meta(out_dir, {"type": "face", "space": "face", "dim": dim or 512,
                        "input": in_name, "output": out_name})
    print(f"  -> {out_dir}")


def make_zip(out_root: str, zip_path: str):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(out_root):
            for name in files:
                p = os.path.join(root, name)
                z.write(p, os.path.relpath(p, out_root))
    print(f"已打包 -> {zip_path}")


def make_face_only_zip(out_root: str, zip_path: str):
    """仅含 FaceEmbedder 的增量包(~十几 MB):设备上已装旧模型包的用户
    导这个即可启用人脸功能,App 端按目录增量导入,不动已有文本/图文模型"""
    face_dir = os.path.join(out_root, "FaceEmbedder")
    if not os.path.isdir(face_dir):
        return
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(face_dir):
            for name in files:
                p = os.path.join(root, name)
                z.write(p, os.path.join("FaceEmbedder", os.path.relpath(p, face_dir)))
    print(f"已打包(仅人脸,增量导入用) -> {zip_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="MediaSeek/Resources/Models")
    ap.add_argument("--gemma", default=GEMMA_DEFAULT)
    ap.add_argument("--siglip", default=SIGLIP_DEFAULT)
    ap.add_argument("--skip-gemma", action="store_true")
    ap.add_argument("--skip-siglip", action="store_true")
    ap.add_argument("--face", default=os.environ.get("FACE_ONNX"),
                    help="人脸识别 ONNX 路径(insightface w600k_mbf.onnx);缺省且环境变量也没给则跳过")
    ap.add_argument("--skip-face", action="store_true")
    ap.add_argument("--fp32", action="store_true", help="不量化,使用 float32(默认 fp16)")
    ap.add_argument("--txt-fp32", action="store_true",
                    help="仅 SigLIP 文本塔用 float32(掩码 fp16 溢出致一致性 0.72 的对策)")
    ap.add_argument("--zip", default=None, help="同时打包为 zip(供 App 内导入)")
    args = ap.parse_args()

    fp16 = not args.fp32
    os.makedirs(args.outdir, exist_ok=True)
    if not args.skip_gemma:
        export_gemma(args.outdir, args.gemma, fp16)
    if not args.skip_siglip:
        export_siglip(args.outdir, args.siglip, fp16, txt_fp32=args.txt_fp32)
    if not args.skip_face and args.face:
        export_face(args.outdir, args.face, fp16)
    elif not args.skip_face:
        print("== 跳过人脸模型(未提供 --face 路径)==")
    if args.zip:
        make_zip(args.outdir, args.zip)
        make_face_only_zip(args.outdir, "face-only.zip")
    print("完成。接下来: xcodegen generate && open MediaSeek.xcodeproj")


if __name__ == "__main__":
    main()
