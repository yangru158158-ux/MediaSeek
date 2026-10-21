#!/usr/bin/env python3
"""SigLIP 开放词表演示(回答「模型对应哪些匹配/哪些英文单词」)。

SigLIP 不是选择题模型(没有固定类目表),是问答题模型:任意英文文本
→文本向量,任意图片→图片向量,余弦相似度打分。本脚本用真实
siglip2-base(与 App 标准模型同款)实测证明:
  1) 任何英文(常用词/冷门词/乱码/整句)都能得到分数——不存在「词表」
  2) 排序有意义:猫图上 cat 最高,海滩图上 beach 最高
  3) 决定搜得好不好的是「分差」(命中词与无关词的间隔),不是「有没有收录」

CPU 可跑;模型约 800MB 首次下载。
"""
import io

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoModel, AutoProcessor

MODEL = "google/siglip2-base-patch16-256"


def fetch(url):
    import urllib.request
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "MediaSeek-dev/1.0 (personal testing; contact via repo)"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return Image.open(io.BytesIO(r.read()))
    except Exception as e:
        print(f"  (图取失败 {url[:60]}…: {e})")
        return None


def load_images():
    cand = {
        "猫照片": ["https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png",
                   "https://commons.wikimedia.org/wiki/Special:FilePath/Cat03.jpg"],
        "狗照片": ["https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"],
        "海滩照片": ["https://commons.wikimedia.org/wiki/Special:FilePath/Maldives%20beach.jpg"],
        "男人照片": ["https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/obama.jpg",
                    "https://commons.wikimedia.org/wiki/Special:FilePath/Albert%20Einstein%20Head.jpg"],
        "女人照片": ["https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
                    "https://commons.wikimedia.org/wiki/Special:FilePath/Mona%20Lisa,%20by%20Leonardo%20da%20Vinci,%20from%20C2RMF%20retouched.jpg"],
    }
    out = {}
    for name, urls in cand.items():
        for u in urls:
            img = fetch(u)
            if img is not None:
                out[name] = img.convert("RGB")
                break
    # 合成截图/文档(与 validate_siglip_coreml.py 同思路)
    scr = Image.new("RGB", (640, 400), "white")
    d = ImageDraw.Draw(scr)
    d.rectangle([0, 0, 639, 48], fill="#1e6fff")
    for i in range(8):
        d.rectangle([24, 80 + i * 36, 600 - (i % 3) * 80, 96 + i * 36], fill="#dfe7ff")
    out["手机截图(合成)"] = scr
    doc = Image.new("RGB", (640, 400), "white")
    d2 = ImageDraw.Draw(doc)
    for i in range(14):
        d2.line([(40, 40 + i * 24), (40 + 500 - (i % 4) * 90, 40 + i * 24)], fill="black", width=2)
    out["文档照片(合成)"] = doc
    return out


TEXTS = [
    "This is a photo of a cat.",
    "This is a photo of a dog.",
    "This is a photo of a beach.",
    "a photo of the ocean at sunset",
    "This is a photo of a woman.",
    "This is a photo of a man.",
    "a young woman smiling",
    "a young man in a suit",
    "This is a photo of a computer screen.",
    "a screenshot of a mobile phone interface",
    "a document with dense text",
    "This is a photo of a machine.",
    "quantum chromodynamics lecture notes",          # 冷门术语
    "a chinchilla eating an ice cream sundae",       # 罕见组合
    "asdf qwerty zxcvbnm",                           # 纯乱码
    "一张女人的照片",                                  # 裸中文(无词典直送)
    "a blurry photo of people having dinner at night",  # 长句
]


def main():
    print(f"加载 {MODEL}(CPU,首次约 800MB)…")
    model = AutoModel.from_pretrained(MODEL).eval()
    proc = AutoProcessor.from_pretrained(MODEL)
    images = load_images()
    print(f"\n图片 {len(images)} 张,查询 {len(TEXTS)} 条\n")

    with torch.no_grad():
        ti = proc(text=TEXTS, padding="max_length", return_tensors="pt", truncation=True)
        # transformers 5:投影层已并入池化,get_*_features 返回带 pooler 的输出
        tvec = model.get_text_features(**ti).pooler_output.numpy()
        ii = proc(images=list(images.values()), return_tensors="pt")
        ivec = model.get_image_features(**ii).pooler_output.numpy()

    def cos(a, b):
        a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    names = list(images)
    print("== 每张图最像的三条查询(证明排序有意义)==")
    for r, name in enumerate(names):
        scores = sorted(((cos(tvec[t], ivec[r]), TEXTS[t]) for t in range(len(TEXTS))), reverse=True)
        print(f"  {name}:")
        for s, q in scores[:3]:
            print(f"    {s:.4f}  {q}")

    print("\n== 关键对照(证明「分差」才是搜索质量,不是「有没有收录」)==")
    def score(q, img):
        t = proc(text=[q], padding="max_length", return_tensors="pt", truncation=True)
        with torch.no_grad():
            v = model.get_text_features(**t).pooler_output.numpy()[0]
        return cos(v, ivec[names.index(img)])

    for img, a, b in [("男人照片", "This is a photo of a man.", "This is a photo of a woman."),
                      ("女人照片", "This is a photo of a woman.", "This is a photo of a man."),
                      ("手机截图(合成)", "a screenshot of a mobile phone interface", "This is a photo of a cat."),
                      ("猫照片", "This is a photo of a cat.", "a document with dense text")]:
        if img not in names:
            print(f"  (跳过 {img}:素材没取到)")
            continue
        sa, sb = score(a, img), score(b, img)
        print(f"  {img}: 命中词「{a}」={sa:.4f}  vs  干扰词「{b}」={sb:.4f}  分差={sa-sb:+.4f}")

    print("\n== 乱码/冷门词也有分数(证明没有「词表」这回事)==")
    for q in ["asdf qwerty zxcvbnm", "quantum chromodynamics lecture notes", "一张女人的照片"]:
        row = [f"{cos(tvec[TEXTS.index(q)], ivec[r]):.4f}" for r in range(len(names))]
        print(f"  「{q}」: " + " ".join(f"{n}={v}" for n, v in zip(names, row)))


if __name__ == "__main__":
    main()
