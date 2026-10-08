#!/usr/bin/env python3
"""给 transformers 5.x 打 coremltools 兼容补丁(幂等):
1) masking_utils.py:掩码组合的按位或 → int32 maximum(布尔等价)
2) embedding_gemma2:多模态占位符掩码 → 恒 False 动态张量(纯文本嵌入场景)
用法: python patch_transformers_for_coreml.py
"""
import os
import sys
from pathlib import Path

import transformers

SP = Path(transformers.__file__).parent
ok, skip = [], []


def patch(path: Path, old: str, new: str, label: str):
    if not path.exists():
        skip.append(f"{label}: 文件不存在 {path.name}")
        return
    s = path.read_text(encoding="utf-8")
    if new in s:
        skip.append(f"{label}: 已打过")
        return
    if old not in s:
        skip.append(f"{label}: 原文未命中(版本差异?)")
        return
    path.write_text(s.replace(old, new), encoding="utf-8", newline="")
    ok.append(label)


# 1) masking_utils 按位或
patch(
    SP / "masking_utils.py",
    "            result = result | mask(batch_idx, head_idx, q_idx, kv_idx).to(result.device)",
    "            result = torch.maximum(result.to(torch.int32), mask(batch_idx, head_idx, q_idx, kv_idx).to(result.device).to(torch.int32)).to(torch.bool)  # patched: | -> int maximum",
    "masking_utils 掩码组合",
)

# 2) embedding_gemma2 多模态掩码(纯文本场景恒 False)
patch(
    SP / "models" / "embedding_gemma2" / "modeling_embedding_gemma2.py",
    "        multimodal_mask = image_mask | video_mask | audio_mask",
    "        multimodal_mask = torch.zeros_like(input_ids, dtype=torch.bool)  # patched: 纯文本嵌入无图/音占位",
    "embedding_gemma2 多模态掩码",
)

print("已打补丁:", ok or "无")
print("跳过:", skip or "无")
if len(ok) + len([s for s in skip if "已打过" in s]) < 2:
    print("⚠️ 有效补丁不足 2 个,请检查 transformers 版本")
    sys.exit(1)
