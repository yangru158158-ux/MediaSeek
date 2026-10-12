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
    from transformers import AutoModel, AutoProcessor, AutoTokenizer

    print(f"== 体检目标 {MODEL_ID} ==")
    model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.float32).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    images = load_test_images()
    print("测试图:", {k: v.size for k, v in images.items()})
    all_texts = sorted({t for ts in TEXTS.values() for t in ts})

    # ---- 显式投影真值(ground truth)
    px_official = {}
    img_gt, txt_gt = {}, {}
    with torch.no_grad():
        ret = model.get_image_features(pixel_values=torch.zeros(1, 3, 256, 256))
        dump_fields("get_image_features", ret)
        ret_t = model.get_text_features(input_ids=torch.ones(1, 8, dtype=torch.int64))
        dump_fields("get_text_features", ret_t)

        for name, img in images.items():
            px = processor(images=img.convert("RGB"), return_tensors="pt")["pixel_values"]
            px_official[name] = px
            img_gt[name] = model.visual_projection(
                model.vision_model(pixel_values=px).pooler_output)[0].numpy()
        for t in all_texts:
            enc = tok(t, return_tensors="pt")
            txt_gt[t] = model.text_projection(
                model.text_model(input_ids=enc["input_ids"],
                                 attention_mask=enc.get("attention_mask")).pooler_output)[0].numpy()

    matched_gt = {n: max(cos(img_gt[n], txt_gt[t]) for t in TEXTS.get(n, [])) for n in images}

    # ---- 真值余弦矩阵
    print("\n== PyTorch 真值(显式投影)匹配对余弦(官方预处理) ==")
    for n in images:
        line = f"  {n.ljust(9)}"
        for t in TEXTS.get(n, []):
            line += f" 「{t[:34]}」={cos(img_gt[n], txt_gt[t]):.4f}"
        print(line)

    # ---- 产线路径转换(与 CI 产线同代码同版本)
    print("\n== 产线路径转换(trace + coremltools fp16) ==")
    with tempfile.TemporaryDirectory() as tmp:
        from export_models import export_siglip, _features_to_tensor
        export_siglip(tmp, MODEL_ID, fp16=True)
        import coremltools as ct
        img_ml = ct.models.MLModel(str(Path(tmp) / "ImageEmbedder" / "SiglipImage.mlpackage"))
        txt_ml = ct.models.MLModel(str(Path(tmp) / "ImageEmbedder" / "SiglipText.mlpackage"))

        def ml_img(px: torch.Tensor) -> np.ndarray:
            v = img_ml.predict({"pixel_values": px.numpy().astype(np.float32)})["embedding"].reshape(-1)
            return v / (np.linalg.norm(v) + 1e-12)

        def ml_txt(ids: list[int]) -> np.ndarray:
            v = txt_ml.predict({"input_ids": np.array([ids], dtype=np.int32)})["embedding"].reshape(-1)
            return v / (np.linalg.norm(v) + 1e-12)

        # 每个候选字段的匹配对余弦——找出"哪一个是真嵌入空间"
        print("\n== 各返回字段 vs 真值/匹配对(以 screen×电脑屏幕、cat×猫 为例) ==")
        with torch.no_grad():
            pxs = px_official["screen"]
            fields = {}
            r = model.get_image_features(pixel_values=pxs)
            if hasattr(r, "keys"):
                for k in r.keys():
                    if torch.is_tensor(r[k]):
                        fields[k] = r[k][0].numpy()
            fields["<显式投影真值>"] = img_gt["screen"]
            t1, t2 = "This is a photo of a computer screen.", "This is a photo of a cat."
            for k, v in fields.items():
                print(f"  screen·{k}: 形状={v.shape} cos(真值)={cos(v, img_gt['screen']):.5f} "
                      f"匹配「{t1[:30]}」={cos(v, txt_gt[t1]):.4f} 交叉「{t2[:22]}」={cos(v, txt_gt[t2]):.4f}")

        # 文本塔:CoreML(产线路径,变长 ids)vs 真值
        print("\n== 文本塔一致性(CoreML 变长 ids vs 显式投影真值) ==")
        for t in all_texts[:4]:
            ids = tok(t)["input_ids"]
            v = ml_txt(ids)
            print(f"  「{t[:36]}」 cos(真值)={cos(v, txt_gt[t]):.5f}")

        # 图像塔:CoreML 官方张量 vs 真值;App 式预处理罚分
        print("\n== 图像塔:CoreML vs 真值(官方张量) + App 式拉伸预处理罚分 ==")
        worst = 1.0
        for name, img in images.items():
            v_off = ml_img(px_official[name])
            consist = cos(v_off, img_gt[name] / np.linalg.norm(img_gt[name]))
            worst = min(worst, consist)
            bi = ml_img(app_style_pixel_values(img, 256, Image.BILINEAR))
            la = ml_img(app_style_pixel_values(img, 256, Image.LANCZOS))
            m = lambda v: max(cos(v, txt_gt[t]) for t in TEXTS.get(name, []))
            print(f"  {name.ljust(9)} cos(真值)={consist:.5f}  官方匹配对={m(v_off):.4f}  "
                  f"拉伸bilinear={m(bi):.4f}  拉伸lanczos={m(la):.4f}")

        # 产线 CoreML 的绝对尺度表(与设备实测 0.088 对质)
        print("\n== 产线 CoreML 官方预处理 匹配对绝对分 vs 设备实测(≈0.088) ==")
        for name in images:
            v = ml_img(px_official[name])
            print(f"  {name.ljust(9)} best={max(cos(v, txt_gt[t]) for t in TEXTS.get(name, [])):.4f}")

    print("\n== 判读 ==")
    print(f"真值匹配对分布: {dict((k, round(v, 4)) for k, v in matched_gt.items())}")
    print(f"产线 CoreML 与显式投影真值最小一致性: {worst:.5f}")
    if worst < 0.98:
        print("H1 定罪:产线工件装的不是显式投影嵌入(_features_to_tensor 抓错字段),修 export_models.py 后重导模型包,App 端全量重嵌")
    else:
        best = max(matched_gt.values())
        if best >= 0.2:
            print(f"H2 嫌疑:真值能打 {best:.3f},对比上方『官方匹配对』与『拉伸』两列判断预处理罚分")
        else:
            print(f"H3:官方预处理下真值匹配对也只有 {best:.3f} → 模型尺度天然低,相对标定正确,升级看 so400m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
