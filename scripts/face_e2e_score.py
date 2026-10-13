#!/usr/bin/env python3
"""r50 端到端打分:对齐图 → onnxruntime w600k_r50 推理 → 余弦矩阵 + 断言。

与 App 完全同款预处理(RGB、(px-127.5)/128、112×112、NCHW)。
命名约定:aligned_p1/p2 = 同人不同照,aligned_p3 = 另一人。
断言语义(相对 App 平衡阈值 0.58 留余量):
  同人 ≥ 0.35 —— 对齐/模型任一环节坏掉时同人分数会掉到 0.1x
  跨人 < 0.45 —— r50 正常跨人 <0.35;≥0.45 说明区分度异常(混人根因在管线)
缺图(下载失败/未检出)自动跳过对应断言;全缺才失败。
"""
import argparse
import glob
import os
import sys

import numpy as np
import onnxruntime as ort
from PIL import Image

SAME_MIN = 0.35
CROSS_MAX = 0.45


def embed(sess, in_name, path):
    img = Image.open(path).convert("RGB").resize((112, 112), Image.BILINEAR)
    x = (np.asarray(img, dtype=np.float32) - 127.5) / 128.0
    x = x.transpose(2, 0, 1)[None]
    out = sess.run(None, {in_name: x})[0].flatten().astype(np.float32)
    return out / (np.linalg.norm(out) + 1e-9)


ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--dir", required=True)
ap.add_argument("--same", default="p1,p2", help="同人照片名缀,逗号分隔")
ap.add_argument("--other", default="p3", help="另一人名缀,逗号分隔")
a = ap.parse_args()

files = sorted(glob.glob(os.path.join(a.dir, "aligned_*.png")))
if not files:
    print("FAIL 无对齐图:检测/对齐环节全失败")
    sys.exit(2)

sess = ort.InferenceSession(a.model, providers=["CPUExecutionProvider"])
in_name = sess.get_inputs()[0].name
vecs = {os.path.basename(f)[len("aligned_"):-len(".png")]: embed(sess, in_name, f)
        for f in files}
names = sorted(vecs)
print("对齐图:", " ".join(names))


def cos(n1, n2):
    return float(np.dot(vecs[n1], vecs[n2]))


for n1 in names:
    print(f"  {n1}: " + " ".join(f"{cos(n1, n2):+.3f}" for n2 in names))

fails = []
same = [s for s in a.same.split(",") if s in vecs]
other = [o for o in a.other.split(",") if o in vecs]
if len(same) >= 2:
    c = cos(same[0], same[1])
    print(f"同人 {same[0]}×{same[1]} = {c:+.3f}(要求 ≥ {SAME_MIN})")
    if c < SAME_MIN:
        fails.append(f"同人相似度过低 {c:.3f} < {SAME_MIN}:对齐或模型环节疑似损坏")
for o in other:
    for s in same:
        c = cos(s, o)
        print(f"跨人 {s}×{o} = {c:+.3f}(要求 < {CROSS_MAX})")
        if c >= CROSS_MAX:
            fails.append(f"跨人相似度过高 {s}×{o} {c:.3f} ≥ {CROSS_MAX}:真实照片上区分度不足")

if fails:
    print("\n".join("FAIL " + f for f in fails))
    sys.exit(1)
print("PASS:真实照片端到端(检测→对齐→r50)区分度正常")
