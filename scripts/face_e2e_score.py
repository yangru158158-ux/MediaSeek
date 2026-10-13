#!/usr/bin/env python3
"""r50 端到端打分 + 聚类/合并验证。

对齐图 → onnxruntime w600k_r50 推理 → 余弦矩阵 → 复刻 App 的两段算法:
  ① 贪心聚类(DuplicateScanner.scan):首个存活样本为锚,≥threshold 入组
  ② 组间合并(DuplicateScanner.mergeGroups):两组任一成员对 ≥linkThreshold 则并组
断言三件:
  同人最低分 ≥ SAME_MIN / 跨人最高分 < CROSS_MAX / **合并后分组 == 真实人物划分**
第三条是 140 用户实测「同一人分了好几组」的回归闸门——前两条过了也可能
因为贪心聚类一锚定终身而把同一人拆开。

文件命名约定:aligned_<人物>_<序号>.png,人物名即真值标签。
"""
import argparse
import glob
import os
import sys
from collections import defaultdict

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


def greedy_cluster(mat, threshold):
    """复刻 DuplicateScanner.scan:首个存活为锚,≥threshold 入组(含单例组)"""
    n = mat.shape[0]
    alive = [True] * n
    groups = []
    while any(alive):
        a = alive.index(True)
        g = [j for j in range(n) if alive[j] and (j == a or mat[a, j] >= threshold)]
        for j in g:
            alive[j] = False
        groups.append(g)
    return groups


def merge_groups(groups, mat, link):
    """复刻 DuplicateScanner.mergeGroups:任一成员对 ≥link 则并组(并查集)"""
    g = len(groups)
    parent = list(range(g))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(g):
        for j in range(i + 1, g):
            if find(i) == find(j):
                continue
            if any(mat[a, b] >= link for a in groups[i] for b in groups[j]):
                ri, rj = find(i), find(j)
                parent[max(ri, rj)] = min(ri, rj)
    buckets = defaultdict(list)
    for i in range(g):
        buckets[find(i)].extend(groups[i])
    return sorted((sorted(v) for v in buckets.values()), key=len, reverse=True)


ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--dir", required=True)
ap.add_argument("--threshold", type=float, default=0.48, help="App 平衡模式入组阈值")
ap.add_argument("--link", type=float, default=0.42, help="组间合并连接阈值(低于入组)")
a = ap.parse_args()

files = sorted(glob.glob(os.path.join(a.dir, "aligned_*.png")))
if not files:
    print("FAIL 无对齐图:检测/对齐环节全失败")
    sys.exit(2)

sess = ort.InferenceSession(a.model, providers=["CPUExecutionProvider"])
in_name = sess.get_inputs()[0].name
names, labels, vecs = [], [], []
for f in files:
    stem = os.path.basename(f)[len("aligned_"):-len(".png")]
    names.append(stem)
    labels.append(stem.rsplit("_", 1)[0])       # 人物名 = 序号前的部分
    vecs.append(embed(sess, in_name, f))
mat = np.stack(vecs) @ np.stack(vecs).T

print("样本:", " ".join(names))
print("相似度矩阵:")
width = max(len(n) for n in names)
for i, n in enumerate(names):
    print(f"  {n:<{width}} " + " ".join(f"{mat[i, j]:+.3f}" for j in range(len(names))))

fails = []
same_pairs = [(i, j) for i in range(len(names)) for j in range(i + 1, len(names))
              if labels[i] == labels[j]]
cross_pairs = [(i, j) for i in range(len(names)) for j in range(i + 1, len(names))
               if labels[i] != labels[j]]
if same_pairs:
    lo = min(mat[i, j] for i, j in same_pairs)
    arg = min(same_pairs, key=lambda p: mat[p[0], p[1]])
    print(f"同人最低 {names[arg[0]]}×{names[arg[1]]} = {lo:+.3f}(要求 ≥ {SAME_MIN})")
    if lo < SAME_MIN:
        fails.append(f"同人相似度过低 {lo:.3f}:对齐或模型环节疑似损坏")
if cross_pairs:
    hi = max(mat[i, j] for i, j in cross_pairs)
    arg = max(cross_pairs, key=lambda p: mat[p[0], p[1]])
    print(f"跨人最高 {names[arg[0]]}×{names[arg[1]]} = {hi:+.3f}(要求 < {CROSS_MAX})")
    if hi >= CROSS_MAX:
        fails.append(f"跨人相似度过高 {hi:.3f}:真实照片上区分度不足")

# 分组验证:贪心聚类 → 组间合并 → 与真值人物划分比对
raw = greedy_cluster(mat, a.threshold)
merged = merge_groups(raw, mat, a.link)


def show(groups):
    return " | ".join("+".join(names[i] for i in g) for g in groups)


print(f"贪心聚类(阈值 {a.threshold}): {show(raw)}")
print(f"组间合并(连接 {a.link}): {show(merged)}")

truth = defaultdict(list)
for i, lb in enumerate(labels):
    truth[lb].append(i)
want = sorted((sorted(v) for v in truth.values()), key=len, reverse=True)
got = sorted((sorted(g) for g in merged), key=len, reverse=True)
print(f"真实人物划分: {show(want)}")
if got != want:
    split = [g for g in want if not any(set(g) <= set(m) for m in got)]
    if split:
        fails.append("同一人被拆成多组(或未完整合并):" + show(split))
    bad = [m for m in got if len({labels[i] for i in m}) > 1]
    if bad:
        fails.append("不同人被合并到一组:" + show(bad))
    if not split and not bad:
        fails.append("分组与真值不一致:" + show(got))
else:
    print("分组 == 真实人物划分 ✓")

if fails:
    print("\n".join("FAIL " + f for f in fails))
    sys.exit(1)
print("PASS:真实照片端到端(检测→对齐→r50→聚类→合并)全部正确")
