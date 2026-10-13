#!/usr/bin/env python3
"""
人脸识别模型导出(insightface MobileFaceNet / ArcFace 系,512 维)→ Core ML。

为什么独立于 export_models.py:coremltools 8/9 已移除内置 ONNX 转换器,
本脚本必须在 coremltools 7.x(最后带 ONNX 支持的版本)的隔离环境里运行:
  python -m venv faceenv
  faceenv/bin/pip install "coremltools==7.2" "onnx==1.16.2" onnxruntime numpy
  faceenv/bin/python scripts/export_face.py --onnx w600k_mbf.onnx --outdir MediaSeek/Resources/Models

产物:
  <outdir>/FaceEmbedder/
    Face.mlpackage   # 输入 input.1 (1,3,112,112) fp16;输出未归一化,App 端统一归一化
    meta.json        # type=face,含 ONNX 原始输入/输出名(如输出 '516')
  face-only.zip      # 仅 FaceEmbedder 的增量包:已装旧模型包的设备导它即启用人脸
  models.zip         # 完整包(若 --outdir 里已有 Text/ImageEmbedder,一并打进)

预处理约定(insightface 官方):RGB、(px-127.5)/128、112×112(ArcFace 对齐由 App 端完成)。
"""
import argparse
import json
import os
import zipfile

import numpy as np
import onnx
import coremltools as ct


def export_face(onnx_path: str, out_root: str, fp16: bool = True):
    print(f"== 导出人脸模型 {onnx_path} ==")
    m = onnx.load(onnx_path)
    in_name = m.graph.input[0].name
    out_name = m.graph.output[0].name
    dim = None
    for d in m.graph.output[0].type.tensor_type.shape.dim:
        if d.dim_value > 1:
            dim = int(d.dim_value)
    print(f"  ONNX 签名: 输入 {in_name} → 输出 {out_name} {dim or '?'} 维")
    print(f"  coremltools {ct.__version__}")

    mlmodel = ct.convert(
        onnx_path,
        inputs=[ct.TensorType(name=in_name, shape=(1, 3, 112, 112), dtype=np.float32)],
        compute_precision=ct.precision.FLOAT16 if fp16 else ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.iOS17,
        convert_to="mlprogram",
    )

    # 一致性对拍:同一随机输入,CoreML vs ONNX 原模型(转换错误的保险丝)
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

    out_dir = os.path.join(out_root, "FaceEmbedder")
    os.makedirs(out_dir, exist_ok=True)
    mlmodel.save(os.path.join(out_dir, "Face.mlpackage"))
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"type": "face", "space": "face", "dim": dim or 512,
                   "input": in_name, "output": out_name}, f, ensure_ascii=False, indent=2)
    print(f"  -> {out_dir}")


def make_zip(out_root: str, zip_path: str, only_face: bool = False):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        if only_face:
            face_dir = os.path.join(out_root, "FaceEmbedder")
            for root, _, files in os.walk(face_dir):
                for name in files:
                    p = os.path.join(root, name)
                    z.write(p, os.path.join("FaceEmbedder", os.path.relpath(p, face_dir)))
        else:
            for root, _, files in os.walk(out_root):
                for name in files:
                    p = os.path.join(root, name)
                    z.write(p, os.path.relpath(p, out_root))
    print(f"已打包{'(仅人脸)' if only_face else ''} -> {zip_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, help="insightface w600k_mbf.onnx 路径")
    ap.add_argument("--outdir", default="MediaSeek/Resources/Models")
    ap.add_argument("--fp32", action="store_true")
    args = ap.parse_args()
    export_face(args.onnx, args.outdir, fp16=not args.fp32)
    make_zip(args.outdir, "face-only.zip", only_face=True)
    make_zip(args.outdir, "models.zip")


if __name__ == "__main__":
    main()
