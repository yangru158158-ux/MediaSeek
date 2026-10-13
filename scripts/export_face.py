#!/usr/bin/env python3
"""
人脸识别模型导出(insightface MobileFaceNet / ArcFace 系,512 维)→ Core ML。

路线:ONNX --onnx2torch--> PyTorch 模块 --trace--> coremltools(ct 9,与主流程同环境)。
为什么不直接 ct.convert(onnx):coremltools 根本没有内置 ONNX 转换器
(7.2 与 9.0 的 converters 目录都只有 libsvm/mil/sklearn/xgboost),
历史上 ONNX 转换走的是独立的 onnx-coreml(只出旧 neuralnetwork 格式,已弃)。

用法(主环境,与 export_models.py 同一个 venv):
  pip install torch "coremltools>=8" onnx onnx2torch onnxruntime numpy
  python scripts/export_face.py --onnx w600k_mbf.onnx --outdir MediaSeek/Resources/Models

产物:
  <outdir>/FaceEmbedder/
    Face.mlpackage   # 输入 face_pixels (1,3,112,112);输出 embedding(未归一化,App 端统一归一化)
    meta.json        # type=face,space=face,dim=512,input/output 名
  face-only.zip      # 仅 FaceEmbedder 的增量包:已装旧模型包的设备导它即启用人脸
  models.zip         # 完整包(--outdir 里已有的 Text/ImageEmbedder 一并打进)

预处理约定(insightface 官方):RGB、(px-127.5)/128、112×112;ArcFace 对齐由 App 端完成。
双重一致性对拍:onnx2torch 转换(torch vs onnxruntime)与 CoreML 转换(coreml vs onnxruntime),
任一 cos < 0.995 直接失败——转换链路上的保险丝。
"""
import argparse
import json
import os
import zipfile

import numpy as np
import onnx
import torch
import coremltools as ct
from onnx2torch import convert as onnx2torch_convert


def export_face(onnx_path: str, out_root: str, fp16: bool = True):
    print(f"== 导出人脸模型 {onnx_path} ==")
    m = onnx.load(onnx_path)
    in_name = m.graph.input[0].name
    out_name = m.graph.output[0].name
    dim = None
    for d in m.graph.output[0].type.tensor_type.shape.dim:
        if d.dim_value > 1:
            dim = int(d.dim_value)
    print(f"  ONNX 签名: {in_name} → {out_name} {dim or '?'} 维;coremltools {ct.__version__}")

    # onnx2torch:ONNX 图 → torch.nn.Module(纯算子包装,不动权重)
    torch_model = onnx2torch_convert(m).eval()
    example = torch.zeros((1, 3, 112, 112), dtype=torch.float32)
    with torch.no_grad():
        traced = torch.jit.trace(torch_model, example)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="face_pixels", shape=(1, 3, 112, 112), dtype=np.float32)],
        outputs=[ct.TensorType(name="embedding")],
        compute_precision=ct.precision.FLOAT16 if fp16 else ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.iOS17,
        convert_to="mlprogram",
    )

    # 双重一致性对拍:同一随机输入,torch/onnxruntime/coreml 三方必须一致
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    rng = np.random.RandomState(0)
    x = rng.rand(1, 3, 112, 112).astype(np.float32)
    ref = sess.run([out_name], {in_name: x})[0].flatten()
    with torch.no_grad():
        t_out = traced(torch.from_numpy(x)).numpy().flatten()
    pred = np.array(mlmodel.predict({"face_pixels": x})["embedding"],
                    dtype=np.float32).flatten()

    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    c_torch = cos(ref, t_out)
    c_ct = cos(ref, pred)
    print(f"  torch↔onnx cos={c_torch:.5f};coreml↔onnx cos={c_ct:.5f}")
    if c_torch < 0.995:
        raise RuntimeError(f"onnx2torch 转换不一致(cos={c_torch:.4f})")
    if c_ct < 0.995:
        raise RuntimeError(f"CoreML 转换一致性不足(cos={c_ct:.4f})")

    out_dir = os.path.join(out_root, "FaceEmbedder")
    os.makedirs(out_dir, exist_ok=True)
    mlmodel.save(os.path.join(out_dir, "Face.mlpackage"))
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"type": "face", "space": "face", "dim": dim or 512,
                   "input": "face_pixels", "output": "embedding",
                   # 模型指纹(ONNX 字节数):App 端比对,换模型自动清空旧脸向量重嵌
                   "model_size": os.path.getsize(onnx_path)}, f,
                  ensure_ascii=False, indent=2)
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
