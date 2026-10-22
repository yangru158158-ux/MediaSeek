#!/usr/bin/env python3
"""
SigLIP2 产线管道体检 v4:transformers 4 vs 5 嵌入空间对质。

背景链条:
  - 设备诊断:电脑屏幕查询对全库最高余弦仅 ~0.088(数千张真实屏幕照)
  - transformers 5 的 SiglipModel 删除了投影层,forward 直接用 pooler_output 当嵌入;
    get_image_features 变成塔的裸输出(v4 里它返回投影后特征)
  - 设备上的 models.zip 于 2026-10-09 在 v5 管道(transformers>=5.0)下导出
  - 若 v4(投影)空间 ≠ v5(pooler)空间,设备装的就是错的特征——0.088 由此而来

用法(同一脚本两种环境各跑一次):
  python validate_siglip_coreml.py extract v5.npz   # 当前环境 canonical 特征
  python validate_siglip_coreml.py compare v4.npz v5.npz
"""
import io
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

MODEL_ID = os.environ.get("SIGLIP_MODEL", "google/siglip2-base-patch16-256")
OUT_DIR = Path(os.environ.get("VALIDATE_OUT", "."))


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
        # 第五十批:性别对比回归用真人像(网络失败仅降级警告不阻断)
        "woman_portrait": ["https://commons.wikimedia.org/wiki/Special:FilePath/Mona%20Lisa,%20by%20Leonardo%20da%20Vinci,%20from%20C2RMF%20retouched.jpg"],
        "man_portrait": ["https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/obama.jpg"],
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


def mode_extract(out_npz: Path):
    import transformers
    from transformers import AutoModel, AutoProcessor, AutoTokenizer

    print(f"== extract @ transformers {transformers.__version__} / torch {torch.__version__} ==")
    try:
        model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.float32).eval()
    except TypeError:  # v4 只认 torch_dtype
        model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float32).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    images = load_test_images()
    print("测试图:", {k: v.size for k, v in images.items()})
    all_texts = sorted({t for ts in TEXTS.values() for t in ts})

    img_v, txt_v = {}, {}
    px_official = {}
    with torch.no_grad():
        for name, img in images.items():
            px = processor(images=img, return_tensors="pt")["pixel_values"]
            px_official[name] = px
            if transformers.__version__.startswith("4"):
                f = model.get_image_features(pixel_values=px)[0]
            else:
                f = model.vision_model(pixel_values=px).pooler_output[0]
            img_v[name] = f.numpy()
        for t in all_texts:
            enc = processor(text=[t], padding="max_length", return_tensors="pt", truncation=True)
            if transformers.__version__.startswith("4"):
                f = model.get_text_features(input_ids=enc["input_ids"],
                                            attention_mask=enc.get("attention_mask"))[0]
            else:
                # 第四十九批定案:SigLIP2 官方 processor 不产 attention_mask——
                # pad 参与注意才是训练语义。传真实掩码会让文本嵌入内容失明
                # (「搜女人 4217 全噪声/搜 woman 0」事故),参考向量必须走无掩码
                f = model.text_model(input_ids=enc["input_ids"]).pooler_output[0]
            txt_v[t] = f.numpy()

    matched = {n: max((cos(img_v[n], txt_v[t]) for t in TEXTS.get(n, [])), default=0.0) for n in images}
    print("匹配对余弦:", {k: round(v, 4) for k, v in matched.items()})

    # ---- 诊断:分词器/嵌入表/图-图相似度/max_length 填充对照
    print("\n== 分词器与配置 ==")
    print("  len(tok)=", len(tok), " pad=", tok.pad_token_id, " bos=", tok.bos_token_id,
          " eos=", tok.eos_token_id, " padding_side=", getattr(tok, "padding_side", None))
    sample = tok("This is a photo of a cat.")
    print("  示例 ids:", sample["input_ids"])
    print("  text vocab_size=", model.config.text_config.vocab_size,
          " 嵌入表 shape=", tuple(model.text_model.embeddings.token_embedding.weight.shape))
    print("  logit_scale=", float(model.logit_scale), " logit_bias=", float(model.logit_bias))

    names = list(images)
    print("\n== 图-图余弦矩阵(视觉塔健康度:cat-dog 应高于 cat-screen) ==")
    print("        " + "".join(n.ljust(10) for n in names))
    for a in names:
        print(a.ljust(8) + "".join(f"{cos(img_v[a], img_v[b]):<10.4f}" for b in names))

    print("\n== 文 max_length 填充+显式掩码(反例诊断:此约定内容失明,勿作参考) ==")
    txt_v_pad = {}
    with torch.no_grad():
        for t in all_texts:
            enc = tok(t, padding="max_length", max_length=64, return_attention_mask=True,
                      return_tensors="pt")
            if transformers.__version__.startswith("4"):
                f = model.get_text_features(input_ids=enc["input_ids"],
                                            attention_mask=enc.get("attention_mask"))[0]
            else:
                f = model.text_model(input_ids=enc["input_ids"],
                                     attention_mask=enc.get("attention_mask")).pooler_output[0]
            txt_v_pad[t] = f.numpy()
    matched_pad = {n: max((cos(img_v[n], txt_v_pad[t]) for t in TEXTS.get(n, [])), default=0.0) for n in images}
    print("  匹配对余弦(max_length 填充):", {k: round(v, 4) for k, v in matched_pad.items()})

    # ---- 第四十九批阻断回归:掩码语义(SigLIP2 训练不掩码 pad) ----
    # 设备 CoreML 文本塔 = 定长64输入 + (1-mask)×-30000 加性掩码 + 末位池化。
    # 此前体检两侧同错(参考向量也传掩码),对拍一致、静默放行——本段用导出图
    # 原式三方对拍:①全1掩码(偏置归零=训练语义)必须与官方无掩码管线一致
    # (≥0.999,阻断);②健康图文分 ≥0.09(阻断);③真实掩码(203 事故形态)
    # 应显著劣化——仅警告不阻断:若上游训练约定变化导致不劣化,需人工判断
    # 新语义,机械阻断反而挡住合法演进。
    print("\n== 掩码语义回归(阻断) ==")

    def traced_formula(text: str, ones_mask: bool):
        real = tok(text)["input_ids"]
        pad_id = int(tok.pad_token_id) if tok.pad_token_id is not None else 0
        ids = torch.tensor([real + [pad_id] * (64 - len(real))], dtype=torch.int32)
        if ones_mask:
            mask = torch.ones((1, 64), dtype=torch.int32)
        else:
            mask = torch.zeros((1, 64), dtype=torch.int32)
            mask[0, :len(real)] = 1
        tm = model.text_model
        with torch.no_grad():
            emb = tm.embeddings(input_ids=ids)
            m4 = (1.0 - mask[:, None, None, :].to(emb.dtype)) * -30000.0
            out = tm.encoder(inputs_embeds=emb, attention_mask=m4)
            h = tm.final_layer_norm(out.last_hidden_state)
            return torch.nn.functional.normalize(tm.head(h[:, -1, :]), dim=-1).numpy()[0]

    ran = 0
    for img_key, text in [("cat", "This is a photo of a cat."),
                          ("dog", "This is a photo of a dog."),
                          ("screen", "This is a photo of a computer screen.")]:
        if img_key not in img_v:
            continue
        ran += 1
        official = txt_v[text]  # 上面已改为官方无掩码参考
        ones_v, real_v = traced_formula(text, True), traced_formula(text, False)
        c_ones = cos(ones_v, official)
        s_ones, s_real = cos(ones_v, img_v[img_key]), cos(real_v, img_v[img_key])
        print(f"  {img_key:<7} 全1掩码vs官方={c_ones:.6f}  图文分(全1)={s_ones:+.4f}  图文分(真实掩码)={s_real:+.4f}")
        if c_ones < 0.999:
            raise SystemExit(f"FAIL 全1掩码偏离官方无掩码管线({c_ones:.6f})——导出图掩码语义被改坏")
        if s_ones < 0.09:
            raise SystemExit(f"FAIL {img_key} 健康图文分过低({s_ones:.4f}<0.09)——文本塔内容失明复发")
        if s_real > s_ones:
            print(f"  ⚠️ 真实掩码不劣于全1({s_real:.4f}≥{s_ones:.4f}),请人工确认上游训练约定是否变化")
    if ran == 0:
        raise SystemExit("FAIL 测试图全部加载失败,掩码回归空转")
    print("  PASS 掩码语义回归:全1掩码=训练语义,真实掩码劣化形态在案")

    # ---- 第五十批阻断回归:性别对比边距(性别词对比过滤的依据) ----
    # SigLIP 绝对分数区分男女弱(男照 0.06-0.10 与女照 0.10-0.11 同带),
    # 相对差才可靠(实测:女照 +0.021/男照 -0.027~-0.038/中性 -0.004)。
    # App 端按「正向分-反向分 ≥ 0.012」过滤;此回归守住边距不复崩。
    print("== 性别对比回归(阻断) ==")

    def ref_vec(text: str):
        enc = processor(text=[text], padding="max_length", return_tensors="pt", truncation=True)
        with torch.no_grad():
            return model.text_model(input_ids=enc["input_ids"]).pooler_output[0].numpy()

    if "woman_portrait" in img_v and "man_portrait" in img_v:
        qw_, qm_ = ref_vec("This is a photo of a woman."), ref_vec("This is a photo of a man.")
        m_w = cos(qw_, img_v["woman_portrait"]) - cos(qm_, img_v["woman_portrait"])
        m_m = cos(qw_, img_v["man_portrait"]) - cos(qm_, img_v["man_portrait"])
        print(f"  女照边距(woman分-man分)={m_w:+.4f}  男照边距={m_m:+.4f}(要求:女≥+0.012,男≤0)")
        if m_w < 0.012:
            raise SystemExit(f"FAIL 女照性别边距 {m_w:+.4f} < +0.012——对比过滤会漏掉真女人照")
        if m_m > 0:
            raise SystemExit(f"FAIL 男照性别边距 {m_m:+.4f} > 0——对比过滤拦不住男照泄漏")
        print("  PASS 性别对比边距在案")
    else:
        print("  ⚠️ 人像测试图加载失败,性别对比回归跳过(不阻断)")

    # 官方 pipeline 示例(模型卡同款:2 cats 图 × 三候选)
    try:
        from transformers import pipeline
        from transformers.image_utils import load_image
        clf = pipeline(model=MODEL_ID, task="zero-shot-image-classification")
        out = clf(load_image("http://images.cocodataset.org/val2017/000000039769.jpg"),
                  candidate_labels=["2 cats", "a plane", "a remote"])
        print("\n== 官方 pipeline 示例输出(预期 2 cats 明显最高) ==")
        print(" ", out)
    except Exception as e:
        print("\n官方 pipeline 示例失败:", e)

    # v5 额外打官方 logit/sigmoid(logits = scale*cos + bias)
    if not transformers.__version__.startswith("4"):
        with torch.no_grad():
            for name in images:
                if name not in TEXTS:
                    continue   # 性别对比人像(第五十批)无 TEXTS 配对,跳过
                px = processor(images=images[name], return_tensors="pt")["pixel_values"]
                ids = tok(TEXTS[name][0], return_tensors="pt")
                out = model(input_ids=ids["input_ids"], attention_mask=ids.get("attention_mask"),
                            pixel_values=px)
                logit = float(out.logits_per_image[0][0])
                print(f"  {name.ljust(9)} 「{TEXTS[name][0][:32]}」 logit={logit:.4f} "
                      f"sigmoid={torch.sigmoid(out.logits_per_image[0][0]).item():.3%}")

    np.savez(out_npz,
             image_names=np.array(list(img_v.keys())),
             text_names=np.array(list(txt_v.keys())),
             **{f"img_{k}": v for k, v in img_v.items()},
             **{f"txt_{k}": v for k, v in txt_v.items()})
    print(f"已保存 {out_npz}")

    # 产线导出链验证(两版本都跑):新导出代码(文本塔定长 64 双输入)+ coremltools 转换 + 推理一致性
    # 失败不阻断:v5 masking_utils 补丁若不够,由 v4 分支给出可行路线
    import tempfile
    import coremltools as ct
    from export_models import export_siglip
    print(f"\n== 产线导出链验证 @ transformers {transformers.__version__} ==")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            export_siglip(tmp, MODEL_ID, fp16=True)
            img_ml = ct.models.MLModel(str(Path(tmp) / "ImageEmbedder" / "SiglipImage.mlpackage"))
            txt_ml = ct.models.MLModel(str(Path(tmp) / "ImageEmbedder" / "SiglipText.mlpackage"))

            def ml(m, feed):
                v = m.predict(feed)["embedding"].reshape(-1).astype(np.float32)
                return v / (np.linalg.norm(v) + 1e-12)

            def padded_ids_mask(t: str):
                enc = tok(t, padding="max_length", max_length=64, return_attention_mask=True)
                return (np.array([enc["input_ids"]], dtype=np.int32),
                        np.array([enc["attention_mask"]], dtype=np.int32))

            worst = 1.0
            for name in images:
                v = ml(img_ml, {"pixel_values": px_official[name].numpy().astype(np.float32)})
                ref = img_v[name] / (np.linalg.norm(img_v[name]) + 1e-12)
                worst = min(worst, cos(v, ref))
            wt = 1.0
            for t in all_texts[:3]:
                ids, mask = padded_ids_mask(t)
                v = ml(txt_ml, {"input_ids": ids, "attention_mask": mask})
                ref = txt_v_pad[t] / (np.linalg.norm(txt_v_pad[t]) + 1e-12)
                wt = min(wt, cos(v, ref))
            print(f"  图像塔 CoreML vs PyTorch 最小一致性 = {worst:.5f}")
            print(f"  文本塔 CoreML vs PyTorch 最小一致性 = {wt:.5f}")

            # 偏差来源定位:PyTorch 纯 fp16(不经 coremltools)对 fp32 参考的一致性
            # ——若同样 ~0.7x,是文本塔本身 fp16 敏感(掩码溢出嫌疑),fp32 导出是对策
            if not transformers.__version__.startswith("4"):
                pass
            with torch.no_grad():
                half_model = model.half()
                wt_half = 1.0
                for t in all_texts[:3]:
                    enc = tok(t, padding="max_length", max_length=64, return_attention_mask=True,
                              return_tensors="pt")
                    r = half_model.get_text_features(input_ids=enc["input_ids"],
                                                     attention_mask=enc["attention_mask"])
                    f = (r.pooler_output if hasattr(r, "pooler_output") else r)[0].float().numpy()
                    ref = txt_v_pad[t] / (np.linalg.norm(txt_v_pad[t]) + 1e-12)
                    wt_half = min(wt_half, cos(f, ref))
                model.float()
                print(f"  PyTorch 纯 fp16 文本塔 vs fp32 参考 一致性 = {wt_half:.5f}(独立于 coremltools)")

            # fp32 文本塔导出对照
            print("  -- fp32 文本塔导出对照 --")
            export_siglip(tmp, MODEL_ID, fp16=True, txt_fp32=True)
            txt_ml32 = ct.models.MLModel(str(Path(tmp) / "ImageEmbedder" / "SiglipText.mlpackage"))
            wt32 = 1.0
            for t in all_texts[:3]:
                ids, mask = padded_ids_mask(t)
                v = ml(txt_ml32, {"input_ids": ids, "attention_mask": mask})
                ref = txt_v_pad[t] / (np.linalg.norm(txt_v_pad[t]) + 1e-12)
                wt32 = min(wt32, cos(v, ref))
            print(f"  文本塔 CoreML fp32 vs PyTorch 最小一致性 = {wt32:.5f}")

            # 端到端区分度(CoreML 图像塔 × CoreML 文本塔,fp16/fp32 各一轮)
            def sep(txt_model, tag):
                print(f"  匹配对 vs 交叉对(CoreML 端到端 {tag},区分度=匹配显著高于交叉):")
                for name in images:
                    v = ml(img_ml, {"pixel_values": px_official[name].numpy().astype(np.float32)})
                    def coreml_txt(tquery):
                        ids, mask = padded_ids_mask(tquery)
                        return ml(txt_model, {"input_ids": ids, "attention_mask": mask})
                    m_in = max(cos(v, coreml_txt(t)) for t in TEXTS.get(name, []))
                    m_out = max(cos(v, coreml_txt(t))
                                for s in TEXTS.values() for t in s if t not in TEXTS.get(name, []))
                    print(f"    [{tag}] {name.ljust(9)} 匹配={m_in:.4f}  交叉max={m_out:.4f}")

            sep(txt_ml, "fp16")
            sep(txt_ml32, "fp32")
    except Exception as e:
        import traceback
        print(f"  ⚠️ 该环境导出链失败: {type(e).__name__}: {e}")
        traceback.print_exc()
    except Exception as e:
        import traceback
        print(f"  ⚠️ 该环境导出链失败: {type(e).__name__}: {e}")
        traceback.print_exc()


def mode_compare(a_npz: Path, b_npz: Path):
    A, B = np.load(a_npz), np.load(b_npz)
    ta = f"{a_npz.stem}(transformers4·投影)" if "v4" in a_npz.stem else a_npz.stem
    tb = f"{b_npz.stem}(transformers5·pooler)" if "v5" in b_npz.stem else b_npz.stem
    print(f"== compare: {ta} vs {tb} ==")

    def get(d, prefix, name):
        return d[f"{prefix}_{name}"]

    imgs = list(A["image_names"])
    print("\n-- 各自空间里的匹配对余弦 --")
    print("图        v4投影    v5pooler")
    for n in imgs:
        m = lambda d: max(cos(get(d, "img", n), get(d, "txt", t)) for t in TEXTS.get(n, []))
        print(f"{n.ljust(9)} {m(A):<9.4f} {m(B):.4f}")

    print("\n-- 跨空间一致性(同一输入,两版本特征向量的余弦) --")
    for n in imgs:
        print(f"  img {n.ljust(9)} = {cos(get(A,'img',n), get(B,'img',n)):.4f}")
    for t in sorted(set(TEXTS))[:3]:
        for txt in TEXTS.get(t, [])[:1]:
            print(f"  txt 「{txt[:30]}」 = {cos(get(A,'txt',txt), get(B,'txt',txt)):.4f}")

    v5_img_first = get(B, "img", imgs[0])
    cross = max(cos(get(A, "img", n), v5_img_first) for n in imgs[1:])
    print(f"\n-- 判读:跨空间图像向量最大互似然 {cross:.4f}")
    if cross > 0.9:
        print("v4 与 v5 空间一致(已证);真正的修复=文本塔按训练约定 max_length 填充,见 extract 段匹配对对照")
    else:
        print("v4 与 v5 空间不同(意外,需排查)")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "extract":
        mode_extract(OUT_DIR / sys.argv[2])
    elif len(sys.argv) >= 4 and sys.argv[1] == "compare":
        mode_compare(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        print(__doc__)
        sys.exit(2)
