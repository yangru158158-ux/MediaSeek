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


def merge_groups(groups, mat, link, expel, rounds=15):
    """复刻 DuplicateScanner.mergeGroups(v3):组质心连接 + 垃圾剔除。

    质心=全组向量均值:同人各子组质心逼近本人均值(照片多的人全并拢),
    模糊/误检脸被平均掉拦不住合并;合并后与组锚 < expel 的成员剔除。
    - 141 单连接版:模糊脸桥把不同人串通(实测 79% 纯度巨组)
    - 142 平均连接版:代表样本挑最低分,大组正确合并被卡死(同人又拆散)
    """
    cur = [list(g) for g in groups]
    X = mat  # 相似度矩阵复用
    for _ in range(rounds):
        n = len(cur)
        if n < 2:
            break
        # 质心两两余弦:用成员均值向量的相似度近似(直接由矩阵聚合)
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                sim = X[np.ix_(cur[i], cur[j])].mean()
                if sim >= link:
                    pairs.append((i, j, float(sim)))
        if not pairs:
            break
        pairs.sort(key=lambda p: -p[2])
        used, nxt, changed = set(), [], False
        for i, j, s in pairs:
            if i in used or j in used:
                continue
            used.update({i, j})
            nxt.append(cur[i] + cur[j])
            changed = True
        for i in range(n):
            if i not in used:
                nxt.append(cur[i])
        cur = nxt
        if not changed:
            break
    out = []
    for g in cur:
        a = g[0]
        keep = [i for i in g if i == a or X[a, i] >= expel]
        if len(keep) >= 2:
            out.append(sorted(keep))
    return sorted(out, key=len, reverse=True)


raw = greedy_cluster(mat, a.threshold)
merged = merge_groups(raw, mat, a.link, a.expel)


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
