#!/usr/bin/env python3
"""SigLIP 查询规范实测(回答「任意英文总得有规范才能理解吧」)。

用与 App 同款 base 模型,把「规范」逐条变成数字:
  ① 模板效应:裸词 vs "This is a photo of X."(44 批修的是什么)
  ② 具体 vs 抽象:像照片说明的句子 vs 抽象概念
  ③ 否定失效:CLIP 系模型不理解 NOT
  ④ 照片类型词/构图词:a screenshot / a close-up 这类限定词真的在起作用
"""
import io
import urllib.request

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoModel, AutoProcessor

MODEL = "google/siglip2-base-patch16-256"


def fetch(url):
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "MediaSeek-dev/1.0 (personal testing)"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return Image.open(io.BytesIO(r.read())).convert("RGB")
    except Exception:
        return None


def synth_screenshot():
    scr = Image.new("RGB", (640, 400), "white")
    d = ImageDraw.Draw(scr)
    d.rectangle([0, 0, 639, 48], fill="#1e6fff")
    for i in range(8):
        d.rectangle([24, 80 + i * 36, 600 - (i % 3) * 80, 96 + i * 36], fill="#dfe7ff")
    return scr


def main():
    model = AutoModel.from_pretrained(MODEL).eval()
    proc = AutoProcessor.from_pretrained(MODEL)
    images = {
        "猫照片": fetch("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png"),
        "女人照片": fetch("https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"),
        "手机截图": synth_screenshot(),
    }
    images = {k: v for k, v in images.items() if v is not None}

    def text_vec(q):
        t = proc(text=[q], padding="max_length", return_tensors="pt", truncation=True)
        with torch.no_grad():
            return model.get_text_features(**t).pooler_output.numpy()[0]

    def img_vec(img):
        ii = proc(images=[img], return_tensors="pt")
        with torch.no_grad():
            return model.get_image_features(**ii).pooler_output.numpy()[0]

    def cos(a, b):
        a, b = a.astype(np.float64), b.astype(np.float64)
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    ivecs = {k: img_vec(v) for k, v in images.items()}

    def row(img, *queries):
        print(f"  [{img}]")
        for q in queries:
            print(f"    {cos(text_vec(q), ivecs[img]):+.4f}  {q!r}")

    print("① 模板效应(裸词被系统性压分——44 批修的就是这个)")
    row("女人照片", "woman", "This is a photo of a woman.")
    row("猫照片", "cat", "This is a photo of a cat.")

    print("\n② 细节三态:写对翻倍 / 写错腰斩(猫照片实况=两只虎斑猫躺码头渔网)")
    row("猫照片",
        "This is a photo of a cat.",
        "This is a photo of two cats.",
        "two tabby cats lying on fishing nets at a harbor",
        "a fluffy cat lying on a couch, close-up photo",
        "the abstract concept of cuteness")
    # 实测:基线 0.1126;数量对 0.1431;细节全对 0.2171(≈×2);
    # 细节错三个(fluffy/couch/close-up)0.0396(腰斩再腰斩)。
    # 结论:细节不是越多越好——写确定对的,不确定就只写主体。

    print("\n③ 否定失效(模型不认识 NOT,分数和不加一样)")
    row("猫照片",
        "This is a photo of a cat.",
        "This is a photo of a dog.",
        "This is a photo of NOT a dog.")

    print("\n④ 照片类型词/构图词(限定词真实起作用)")
    row("手机截图",
        "a screenshot of a mobile app interface",
        "a photo taken outdoors with a camera",
        "a close-up photo of a flower")


if __name__ == "__main__":
    main()
