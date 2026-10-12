#!/usr/bin/env python3
"""
SigLIP2 产线管道体检 v2(在 macOS CI 上运行,与 export_models.yml 同环境同依赖)。

背景:设备端诊断「This is a photo of a computer screen.」对全库最高余弦仅 ~0.088。
第一轮体检发现:transformers 5 的 get_image_features/get_text_features 返回 ModelOutput,
首字段是未池化 last_hidden_state;产线 _features_to_tensor 优先抓 pooler_output——
若 ModelOutput 里的 pooler_output 是未过投影层的池化状态,图文两侧都变成"未投影特征",
余弦必然 quasi-random 0.05-0.1,与设备实测完全吻合。本轮直接定罪:
  A) 枚举 get_*_features 返回对象的每个字段,逐字段算图文匹配对余弦
  B) 产线路径转换出的 CoreML 模型 vs 显式投影真值(visual/text_projection(pooler_output))
     → 判定产线工件里装的是"真嵌入"还是"未投影状态"
判读:
  - CoreML≈真值(cos>0.98):产线装的是真嵌入 → 看 H2 预处理/H3 模型尺度
  - CoreML≈某个未投影字段而≠真值:H1 定罪,修 export_models 的 _features_to_tensor
"""
import io
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))

MODEL_ID = os.environ.get("SIGLIP_MODEL", "google/siglip2-base-patch16-256")


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
    img = Image.new("RGB", (1179, 900), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 1179, 90], fill=(43, 87, 154))
    d.rectangle([0, 90, 220, 900], fill=(232, 232, 232))
    d.rectangle([40, 30, 300, 62], fill=(255, 255, 255))
    y = 140
    for w, h in [(860, 36), (780, 36), (820, 36), (640, 36), (860, 120),
                 (500, 36), (790, 36), (300, 36), (860, 90), (720, 36)]:
        d.rectangle([260, y, 260 + w, y + h], fill=(30, 30, 30))
        y += h + 34
    return img


def synth_document() -> Image.Image:
    img = Image.new("RGB", (850, 1100), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([60, 50, 700, 96], fill=(20, 20, 20))
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
                out[name] = img.convert("RGB")
                break
    out["screen"] = synth_screenshot()
    out["document"] = synth_document()
    return out


TEXTS = {
    "screen": ["This is a photo of a computer screen.", "a computer screen"],
    "cat": ["This is a photo of a cat.", "a cat"],
    "dog": ["This is a photo of a dog.", "a dog"],
    "beach": ["This is a photo of a beach.", "a tropical beach"],
    "document": ["a document with text"],
}


def cos(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def app_style_pixel_values(img: Image.Image, size: int, resample) -> torch.Tensor:
    im = img.convert("RGB").resize((size, size), resample)
    x = np.asarray(im).astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    return torch.from_numpy(x.transpose(2, 0, 1))[None]


def dump_fields(tag: str, ret):
    print(f"-- {tag} 返回类型: {type(ret).__name__}")
    if hasattr(ret, "keys"):
        for k in ret.keys():
            v = ret[k]
            print(f"     {k}: {tuple(v.shape) if torch.is_tensor(v) else type(v)}")


def main():
    import torch.nn.functional as F
    from transformers import AutoModel, AutoProcessor, AutoTokenizer

    print(f"== 体检目标 {MODEL_ID} ==")
    model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.float32).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    images = load_test_images()
    print("测试图:", {k: v.size for k, v in images.items()})
    all_texts = sorted({t for ts in TEXTS.values() for t in ts})

    # 模型结构速览(找投影层改名后的位置)
    print("model 子模块:", [n for n, _ in model.named_children()])

    # ---- 真值:完整 forward 的 image_embeds / text_embeds(联合嵌入空间定义)
    px_official = {}
    img_gt, txt_gt = {}, {}
    with torch.no_grad():
        probe = model(input_ids=torch.ones(1, 8, dtype=torch.int64))
        dump_fields("forward(input_ids)", probe)
        for name, img in images.items():
            px = processor(images=img.convert("RGB"), return_tensors="pt")["pixel_values"]
            px_official[name] = px
            img_gt[name] = model(pixel_values=px).image_embeds[0].numpy()
        for t in all_texts:
            enc = tok(t, return_tensors="pt")
            txt_gt[t] = model(input_ids=enc["input_ids"],
                              attention_mask=enc.get("attention_mask")).text_embeds[0].numpy()

    matched_gt = {n: max(cos(img_gt[n], txt_gt[t]) for t in TEXTS.get(n, [])) for n in images}

    # ---- 真值余弦 vs 产线实际抓的 pooler_output(未投影)
    print("\n== 真值(forward embeds)匹配对余弦 vs 产线 pooler_output 路径 ==")
    print("  图        真值best   pooler_output(产线现役)")
    for n in images:
        with torch.no_grad():
            pool = model.vision_model(pixel_values=px_official[n]).pooler_output[0].numpy()
        print(f"  {n.ljust(9)} {matched_gt[n]:<10.4f} "
              f"{max(cos(pool, txt_gt[t]) for t in TEXTS.get(n, [])):.4f}")

    # ---- 两条 CoreML 路径对质
    print("\n== CoreML 对质:产线路径(pooler_output) vs 修正路径(forward embeds) ==")
    with tempfile.TemporaryDirectory() as tmp:
        import coremltools as ct

        common = dict(
            outputs=[ct.TensorType(name="embedding")],
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.iOS17,
            convert_to="mlprogram",
        )
        # 产线路径原样复刻
        class ProdImage(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.m = model
            def forward(self, pixel_values):
                return F.normalize(_features_to_tensor(self.m.get_image_features(pixel_values=pixel_values)), dim=-1)

        class FixedImage(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.m = model
            def forward(self, pixel_values):
                return F.normalize(self.m(pixel_values=pixel_values).image_embeds, dim=-1)

        class FixedText(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.m = model
            def forward(self, input_ids):
                return F.normalize(self.m(input_ids=input_ids).text_embeds, dim=-1)

        ex = torch.zeros(1, 3, 256, 256)
        with torch.no_grad():
            prod_ml = ct.convert(torch.jit.trace(ProdImage().eval(), ex),
                                 inputs=[ct.TensorType(name="pixel_values", shape=(1, 3, 256, 256), dtype=np.float32)], **common)
            fixed_img_ml = ct.convert(torch.jit.trace(FixedImage().eval(), ex),
                                      inputs=[ct.TensorType(name="pixel_values", shape=(1, 3, 256, 256), dtype=np.float32)], **common)
            fixed_txt_ml = ct.convert(torch.jit.trace(FixedText().eval(), torch.ones(1, 16, dtype=torch.int32)),
                                      inputs=[ct.TensorType(name="input_ids", shape=(1, ct.RangeDim(1, 64)), dtype=np.int32)], **common)

        def ml(m, feed):
            v = m.predict(feed)["embedding"].reshape(-1)
            return v / (np.linalg.norm(v) + 1e-12)

        print("\n== 文本塔一致性(CoreML 修正路径 vs 真值) ==")
        for t in all_texts[:4]:
            ids = tok(t)["input_ids"]
            print(f"  「{t[:36]}」 cos(真值)={cos(ml(fixed_txt_ml, {'input_ids': np.array([ids], dtype=np.int32)}), txt_gt[t]):.5f}")

        print("\n图".rjust(0) + "\n== 图像塔一致性 + 匹配对尺度 ==")
        print("  图        产线cos(真值)  修正cos(真值)  修正匹配对best")
        worst_fixed = 1.0
        for name, img in images.items():
            v_prod = ml(prod_ml, {"pixel_values": px_official[name].numpy().astype(np.float32)})
            v_fix = ml(fixed_img_ml, {"pixel_values": px_official[name].numpy().astype(np.float32)})
            c_fix = cos(v_fix, img_gt[name] / np.linalg.norm(img_gt[name]))
            worst_fixed = min(worst_fixed, c_fix)
            best_fix = max(cos(v_fix, txt_gt[t]) for t in TEXTS.get(name, []))
            print(f"  {name.ljust(9)} {cos(v_prod, img_gt[name]/np.linalg.norm(img_gt[name])):<14.4f} "
                  f"{c_fix:<14.5f} {best_fix:.4f}")

        # App 式拉伸预处理罚分(修正路径)
        print("\n== 修正路径:官方预处理 vs App 式拉伸预处理 匹配对 ==")
        for name, img in images.items():
            bi = ml(fixed_img_ml, {"pixel_values": app_style_pixel_values(img, 256, Image.BILINEAR).numpy().astype(np.float32)})
            m_off = max(cos(ml(fixed_img_ml, {"pixel_values": px_official[name].numpy().astype(np.float32)}), txt_gt[t]) for t in TEXTS.get(name, []))
            m_bi = max(cos(bi, txt_gt[t]) for t in TEXTS.get(name, []))
            print(f"  {name.ljust(9)} 官方={m_off:.4f}  拉伸bilinear={m_bi:.4f}")

    print("\n== 判读 ==")
    print(f"真值匹配对分布: {dict((k, round(v, 4)) for k, v in matched_gt.items())}")
    print(f"修正路径 CoreML 与真值最小一致性: {worst_fixed:.5f} (≥0.98 即修正方案可上线)")
    print("产线现役(pooler_output)与真值的余弦见上表——接近 0 即坐实『产线装的是未投影特征』")
    return 0


if __name__ == "__main__":
    sys.exit(main())
