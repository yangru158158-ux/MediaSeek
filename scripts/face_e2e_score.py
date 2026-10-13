#!/usr/bin/env python3
"""r50 端到端打分 + 聚类/合并验证。

对齐图 → onnxruntime w600k_r50 推理 → 余弦矩阵 → 复刻 App 的两段算法:
  ① 贪心聚类(DuplicateScanner.scan):首个存活样本为锚,≥threshold 入组
  ② 组间合并(DuplicateScanner.mergeGroups v3):组质心连接 + 垃圾剔除
断言:
  同人最低分 ≥ SAME_MIN / 跨人最高分 < CROSS_MAX /
  合并后分组 == 真实人物划分 / 组内成员对锚 ≥ 剔除线
后两条是回归闸门——140「同一人分了好几组」、141「-6% 垃圾脸混组」都会红。

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


def merge_groups(groups, mat, link, expel, absorb=None, rounds=25):
    """复刻 DuplicateScanner.mergeGroups(v4):组质心连接 + 离群吸收 + 垃圾剔除。

    v3 只合并贪心组,离群照片(够不到任何锚)永远单例;v4 让单例并入质心
    最像的组。真实数据同人最低 0.48/跨人最高 0.16,中间 0.2~0.45 空带——
    连接/吸收阈值放空带下段,跨人不可能串组,模糊桥由末端剔除兜底。
    """
    if absorb is None:
        absorb = link - 0.03
    cur = [list(g) for g in groups if len(g) > 1]
    sng = [g[0] for g in groups if len(g) == 1]
    for _ in range(rounds):
        n = len(cur)
        if n < 2:
            break
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                sim = float(mat[np.ix_(cur[i], cur[j])].mean())
                if sim >= link:
                    pairs.append((i, j, sim))
        if not pairs:
            break
        pairs.sort(key=lambda p: -p[2])
        used, nxt = set(), []
        for i, j, s in pairs:
            if i in used or j in used:
                continue
            used.update({i, j})
            nxt.append(cur[i] + cur[j])
        for i in range(n):
            if i not in used:
                nxt.append(cur[i])
        cur = nxt
    # 离群吸收:单例并入质心(均值相似度近似)最像的组;单例间不互并
    for s in sng:
        sims = [float(mat[s, g].mean()) for g in cur]
        gi = int(np.argmax(sims)) if sims else -1
        if gi >= 0 and sims[gi] >= absorb:
            cur[gi].append(s)
    out = []
    for g in cur:
        a = g[0]
        keep = [i for i in g if i == a or mat[a, i] >= expel]
        if len(keep) >= 2:
            out.append(sorted(keep))
    return out


ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--dir", required=True)
ap.add_argument("--threshold", type=float, default=0.48, help="App 平衡模式入组阈值")
ap.add_argument("--link", type=float, default=0.30, help="组间质心连接阈值(空带下段)")
ap.add_argument("--absorb", type=float, default=0.27, help="离群照片吸收线")
ap.add_argument("--expel", type=float, default=0.18, help="垃圾剔除线:与组锚相似低于则移出组")
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
width = max(len(n) for n in names)
for i, n in enumerate(names):
    print(f"  {n:<{width}} " + " ".join(f"{mat[i, j]:+.3f}" for j in range(len(names))))

fails = []
same_pairs = [(i, j) for i in range(len(names)) for j in range(i + 1, len(names))
              if labels[i] == labels[j]]
cross_pairs = [(i, j) for i in range(len(names)) for j in range(i + 1, len(names))
               if labels[i] != labels[j]]
if same_pairs:
    (i, j) = min(same_pairs, key=lambda p: mat[p[0], p[1]])
    lo = float(mat[i, j])
    print(f"同人最低 {names[i]}×{names[j]} = {lo:+.3f}(要求 ≥ {SAME_MIN})")
    if lo < SAME_MIN:
        fails.append(f"同人相似度过低 {lo:.3f}:对齐或模型环节疑似损坏")
if cross_pairs:
    (i, j) = max(cross_pairs, key=lambda p: mat[p[0], p[1]])
    hi = float(mat[i, j])
    print(f"跨人最高 {names[i]}×{names[j]} = {hi:+.3f}(要求 < {CROSS_MAX})")
    if hi >= CROSS_MAX:
        fails.append(f"跨人相似度过高 {hi:.3f}:真实照片上区分度不足")

raw = greedy_cluster(mat, a.threshold)
merged = merge_groups(raw, mat, a.link, a.expel, a.absorb)


def show(groups):
    return " | ".join("+".join(names[i] for i in g) for g in groups)


print(f"贪心聚类(阈值 {a.threshold}): {show(raw)}")
print(f"组间合并(质心连接 {a.link} 剔除 {a.expel}): {show(merged)}")

# 垃圾剔除闸门:组内任何成员对组锚(g[0])的相似度不得低于剔除线
# (141 实测「最弱 -6%」=模糊/误检脸混进大组,v3 应当已把它们剔出)
for g in merged:
    if len(g) < 2:
        continue
    low = min(mat[g[0], i] for i in g[1:])
    if low < a.expel - 1e-6:
        fails.append(f"组内残留垃圾成员 对锚 {low:+.3f} < {a.expel}:" + show([g]))

truth = defaultdict(list)
for i, lb in enumerate(labels):
    truth[lb].append(i)
want = {frozenset(v) for v in truth.values()}
got = {frozenset(g) for g in merged}
print("真实人物划分:", show(sorted((sorted(s) for s in want), key=len, reverse=True)))
if got != want:
    split = [g for g in want if not any(g <= m for m in got)]
    if split:
        fails.append("同一人被拆成多组(或未完整合并):"
                     + show(sorted((sorted(s) for s in split), key=len, reverse=True)))
    bad = [m for m in got if len({labels[i] for i in m}) > 1]
    if bad:
        fails.append("不同人被合并到一组:"
                     + show(sorted((sorted(s) for s in bad), key=len, reverse=True)))
    if not split and not bad:
        fails.append("分组与真值不一致")
else:
    print("分组 == 真实人物划分 ✓")

if fails:
    print("\n".join("FAIL " + f for f in fails))
    sys.exit(1)
print("PASS:真实照片端到端(检测→对齐→r50→聚类→合并)全部正确")
