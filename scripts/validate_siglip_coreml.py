#!/usr/bin/env python3
"""
SigLIP2 产线管道体检(在 macOS CI 上运行,与 export_models.yml 同环境同依赖)。

背景:设备端诊断显示「This is a photo of a computer screen.」对全库(含数千张真实屏幕照)
最高余弦仅 ~0.088,强查询与弱查询(0.068-0.090)几乎分不开。三种假设:
  H1 转换/trace 有 bug    → CoreML 输出与 PyTorch 参考余弦应 <0.98
  H2 App 预处理有罚分     → 官方预处理分数正常,App 式拉伸预处理分数明显低
  H3 模型尺度天然就低     → 官方预处理下匹配对也只有 ~0.1(相对标定路线正确)

方法:
  1) 下载/生成测试图(猫/狗/合成截图/合成文档/海滩)
  2) PyTorch 参考:官方 AutoProcessor 预处理 → 全量图文余弦矩阵(尺度真相)
  3) 产线转换:直接 import export_models.export_siglip(与 CI 产线逐字节同路径)
  4) 一致性:同一官方预处理张量喂 PyTorch 与 CoreML,逐图算余弦
  5) 预处理罚分:App 式 PIL 拉伸(bilinear/lanczos)+ (x/255-0.5)/0.5 过 CoreML,对比官方预处理
"""
import os
import sys
import io
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))

MODEL_ID = os.environ.get("SIGLIP_MODEL", "google/siglip2-base-patch16-256")

# ---------------------------------------------------------------- 测试图

def fetch(url: str) -> Image.Image | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 medseek-validate"})
        raw = urllib.request.urlopen(req, timeout=30).read()
        img = Image.open(io.BytesIO(raw))
        img.load()
        return img
    except Exception as e:
        print(f"  下载失败 {url}: {e}")
        return None


def synth_screenshot() -> Image.Image:
    """合成一张典型应用截图:蓝标题栏+灰侧栏+黑色文字条(白底)。"""
    img = Image.new("RGB", (1179, 900), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 1179, 90], fill=(43, 87, 154))          # 标题栏
    d.rectangle([0, 90, 220, 900], fill=(232, 232, 232))       # 侧栏
    d.rectangle([40, 30, 300, 62], fill=(255, 255, 255))       # 标题文字位
    y = 140
    for w, h in [(860, 36), (780, 36), (820, 36), (640, 36), (860, 120),
                 (500, 36), (790, 36), (300, 36), (860, 90), (720, 36)]:
        d.rectangle([260, y, 260 + w, y + h], fill=(30, 30, 30))
        y += h + 34
    return img


def synth_document() -> Image.Image:
    """合成一页竖版文档:白底+黑字行+深色标题。"""
    img = Image.new("RGB", (850, 1100), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([60, 50, 700, 96], fill=(20, 20, 20))          # 标题
    y = 160
    rng = np.random.RandomState(7)
    while y < 1020:
        w = int(rng.uniform(0.5, 0.95) * 720)
        d.rectangle([70, y, 70 + w, y + 22], fill=(40, 40, 40))
        y += 48
    return img


def load_test_images() -> dict[str, Image.Image]:
    out: dict[str, Image.Image] = {}
    for name, urls in {
        "cat": ["https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png",
                "https://commons.wikimedia.org/wiki/Special:FilePath/Cat03.jpg"],
        "dog": ["https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"],
        "beach": ["https://commons.wikimedia.org/wiki/Special:FilePath/Maldives%20beach.jpg",
                  "https://commons.wikimedia.org/wiki/Special:FilePath/Beach%20of%20Varkala,%20Kerala.jpg"],
    }.items():
        for u in urls:
            img = fetch(u)
            if img is not None:
                out[name] = img
                break
    out["screen"] = synth_screenshot()
    out["document"] = synth_document()
    return out


TEXTS = {
    "screen": ["This is a photo of a computer screen.", "a computer screen", "a photo of a passport"],
    "cat": ["This is a photo of a cat.", "a cat"],
    "dog": ["This is a photo of a dog.", "a dog"],
    "beach": ["This is a photo of a beach.", "a tropical beach"],
    "document": ["a document with text", "a photo of a computer screen"],
}


# ---------------------------------------------------------------- 预处理

def official_pixel_values(processor, img: Image.Image) -> torch.Tensor:
    out = processor(images=img, return_tensors="pt")
    return out["pixel_values"]


def app_style_pixel_values(img: Image.Image, size: int, resample) -> torch.Tensor:
    """App 端 ImagePreprocess.normalizedNCHW 的等价 Python 实现:直接拉伸到方图。"""
    im = img.convert("RGB").resize((size, size), resample)
    x = np.asarray(im).astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    return torch.from_numpy(x.transpose(2, 0, 1))[None]


def cos(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# ---------------------------------------------------------------- 主流程

def main():
    from transformers import AutoModel, AutoProcessor, AutoTokenizer

    print(f"== 体检目标 {MODEL_ID} ==")
    model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.float32).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    pconf = getattr(processor, "image_processor", processor)
    print("官方图像预处理配置:", {k: getattr(pconf, k, None) for k in
          ("size", "crop_size", "resample", "do_resize", "do_center_crop", "image_mean", "image_std")})

    images = load_test_images()
    print("测试图:", {k: v.size for k, v in images.items()})
    assert images, "没有任何测试图可用"

    all_texts = sorted({t for ts in TEXTS.values() for t in ts})
    text_emb_pt = {}
    for t in all_texts:
        ids = tok(t, return_tensors="pt")  # HF 默认按 tokenizer 配置加特殊 token
        with torch.no_grad():
            f = model.get_text_features(**{k: v for k, v in ids.items() if k in ("input_ids", "attention_mask")})
        text_emb_pt[t] = f[0].numpy()
        text_emb_pt[t + "|NOMASK"] = model.get_text_features(input_ids=ids["input_ids"])[0].detach().numpy()

    # --- 2) PyTorch 参考:官方预处理
    print("\n== PyTorch 参考余弦矩阵(官方预处理,行=图,列=文)==")
    pt_emb, px_official = {}, {}
    for name, img in images.items():
        px = official_pixel_values(processor, img)
        px_official[name] = px
        with torch.no_grad():
            f = model.get_image_features(pixel_values=px)
        pt_emb[name] = f[0].numpy()
    header = "图\\文".ljust(10) + "".join(t[:26].ljust(28) for t in all_texts)
    print(header)
    for name in images:
        row = name.ljust(10) + "".join(f"{cos(pt_emb[name], text_emb_pt[t]):<28.4f}" for t in all_texts)
        print(row)

    matched_ref = {n: max(cos(pt_emb[n], text_emb_pt[t]) for t in TEXTS.get(n, [])) for n in images}
    print("\n各图最优匹配对(PyTorch/官方预处理):",
          {k: round(v, 4) for k, v in matched_ref.items()})

    # --- 3) 产线路径转换(import export_models,与 CI 产线同代码)
    print("\n== 产线路径转换(trace + coremltools fp16)==")
    with tempfile.TemporaryDirectory() as tmp:
        from export_models import export_siglip
        export_siglip(tmp, MODEL_ID, fp16=True)
        import coremltools as ct
        img_ml = ct.models.MLModel(str(Path(tmp) / "ImageEmbedder" / "SiglipImage.mlpackage"))
        txt_ml = ct.models.MLModel(str(Path(tmp) / "ImageEmbedder" / "SiglipText.mlpackage"))

        def ml_embed(m, feed):
            v = m.predict(feed)["embedding"].reshape(-1).astype(np.float32)
            return v / (np.linalg.norm(v) + 1e-12)

        # 文本塔一致性(产线路径=不填充定长,直接变长 ids)
        print("\n== 文本塔一致性(CoreML 变长 ids vs PyTorch 同 ids)==")
        for t in all_texts[:4]:
            ids = tok(t)["input_ids"]
            arr = np.array([ids], dtype=np.int32)
            v = ml_embed(txt_ml, {"input_ids": arr})
            ref = model.get_text_features(input_ids=torch.tensor(arr))[0].detach().numpy()
            ref_n = ref / (np.linalg.norm(ref) + 1e-12)
            print(f"  「{t[:40]}」 一致性 cos={cos(v, ref_n):.5f}  产线输出与官方模板文本的余弦="
                  f"{max(cos(v, text_emb_pt[x]) for x in all_texts):.4f}")

        # --- 4) 图像塔一致性 + 5) 预处理罚分
        from PIL import Image as PImage
        print("\n== 图像塔:一致性(官方张量) 与 App 式预处理罚分 ==")
        print("图".ljust(10) + "一致性cos".ljust(12) + "官方@CoreML".ljust(13) + "拉伸bilinear".ljust(13) + "拉伸lanczos")
        worst = 1.0
        for name, img in images.items():
            px = px_official[name]
            ref_n = pt_emb[name] / (np.linalg.norm(pt_emb[name]) + 1e-12)
            coreml_official = ml_embed(img_ml, {"pixel_values": px.numpy().astype(np.float32)})
            consist = cos(coreml_official, ref_n)
            worst = min(worst, consist)
            bi = ml_embed(img_ml, {"pixel_values": app_style_pixel_values(img, px.shape[-1], PImage.BILINEAR).numpy().astype(np.float32)})
            la = ml_embed(img_ml, {"pixel_values": app_style_pixel_values(img, px.shape[-1], PImage.LANCZOS).numpy().astype(np.float32)})
            def matched(v):
                return max(cos(v, text_emb_pt[t]) for t in TEXTS.get(name, []))
            print(name.ljust(10) + f"{consist:<12.5f}{matched(coreml_official):<13.4f}{matched(bi):<13.4f}{matched(la):.4f}")

    # --- 判读
    print("\n== 判读 ==")
    if worst < 0.98:
        print(f"H1 成立嫌疑大:最小一致性 {worst:.4f} < 0.98 → trace/coremltools 转换有 bug,先修转换再谈换模型")
    else:
        print(f"转换一致性通过(最小 {worst:.4f} ≥ 0.98)→ CoreML 与 PyTorch 一致,H1 排除")
    best_ref = max(matched_ref.values())
    print(f"官方预处理下最优匹配对 = {best_ref:.4f}:"
          + ("模型尺度天然低(H3),107/118 相对标定路线正确,提升区分度需换更大模型(so400m)"
             if best_ref < 0.15 else
             "模型本身能打出高分(H2 嫌疑):对比 App 式两列,若明显低于官方列 → App 预处理是元凶,修 ImagePreprocess 即可"))
    print("\n(产线 App 端此前设备实测:电脑屏幕全库最高 ≈0.088)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
