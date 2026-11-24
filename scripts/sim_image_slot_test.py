#!/usr/bin/env python3
"""2.0 图像模型槽位迁移与导入文件视觉向量重建的状态机模拟测试
(替换第四十六批的 sim_model_switch_test.py——那套「标准 ↔ so400m 切换」逻辑
已随 Base 停用整段删除)。

本机编译不了 ModelManager/IndexingCoordinator/Views(不在 swiftc 本地类型检查集),
把 2.0 改动的控制流**忠实移植**成 Python 状态机,在模拟的文件系统/UserDefaults/
通知/向量库上跑全部场景断言。⚠️ 这是逻辑镜像不是 Swift 编译验证:类型错仍要等
CI;Swift 代码为准,改 Swift 必同步改这里。

镜像的 Swift 逻辑(括号内为 Swift 函数与所在文件):
  migrate_image_slot   (migrateImageSlot,ModelManager)        启动迁移,纯目录状态驱动
  load_clip            (loadSlot 的 "clip" 分支,ModelManager)  「不用 Base」的最终守卫+维度变更广播
  import_zip           (importModelZip,ModelManager)          目录名即槽位,Base 目录跳过
  installed_fingerprint(installedFingerprint,ModelManager)    与 export_models.yml 的清单指纹同规
  wait_ready           (waitReady,ModelManager)               图像模型没装时不干等满超时
  sync_imported_files  (syncImportedFiles,IndexingCoordinator) 导入文件的视觉向量按模型维度重建

场景编号:S*=槽位迁移(对应 2.0 方案记忆里的迁移状态机),F*=导入文件视觉向量重建。
直接跑: python scripts/sim_image_slot_test.py
"""
import json
import sys

MIN_DIM = 1024                       # ModelManager.minImageModelDim
BASE = "ImageEmbedder"               # ModelSlot.image.rawValue
PRO = "ImageEmbedderPro"             # 1.0 的 so400m 槽位(2.0 已无此枚举值)
META_BASE = {"type": "clip", "space": "clip", "dim": 768, "image_size": 256}
META_PRO = {"type": "clip", "space": "clip", "dim": 1152, "image_size": 384}
META_TEXT = {"type": "gemma", "space": "gemma", "dim": 512}
META_FACE = {"type": "face", "space": "face", "dim": 512, "model_size": 174383860}
# 在线源旧清单实测值(2026-11-24 Range 探针):已装 1.0 全套模型的设备指纹
OLD_FULL_FP = "T:512|I:768|F:512:174383860"
COMPILED = ("SiglipImage.mlmodelc", "SiglipText.mlmodelc")   # loadSlot 加载的两个目录
VISUAL_EXT = (".jpg", ".jpeg", ".png", ".heic", ".mp4", ".mov")   # 图片/视频(UTType image/movie)


def full(meta):
    """完整的模型目录:meta.json + 两个已编译模型目录"""
    return dict({"meta.json": json.dumps(meta)}, **{n: "<mlmodelc>" for n in COMPILED})


def half(meta):
    """半成品目录:1.0 导入中途失败留下的「有 meta.json、没 mlmodelc」"""
    return {"meta.json": json.dumps(meta)}


def is_visual(name):
    return name.lower().endswith(VISUAL_EXT)


# ---------------------------------------------------------------- 模拟基座

class FS:
    """Application Support/MediaSeek/Models/:目录名 -> {文件名: 内容}"""
    def __init__(self, dirs=None):
        self.dirs = {k: dict(v) for k, v in (dirs or {}).items()}
        self.rmtree_fail = set()     # 注入「目录删不掉」
        self.move_fail = False       # 注入「改名抛错」

    def put(self, d, meta):
        self.dirs[d] = full(meta)

    def exists(self, d):
        return d in self.dirs

    def rm(self, d):
        if d not in self.dirs:
            return True              # 不存在视为已清
        if d in self.rmtree_fail:
            return False
        del self.dirs[d]
        return True

    def move(self, src, dst):
        if self.move_fail:
            raise OSError("moveItem failed")
        assert src in self.dirs and dst not in self.dirs
        self.dirs[dst] = self.dirs.pop(src)

    def dim(self, d):
        """对应 storedDim(_:):meta 读不出/解不出返回 None"""
        raw = self.dirs.get(d, {}).get("meta.json")
        if raw is None:
            return None
        try:
            return json.loads(raw).get("dim")
        except ValueError:
            return None

    def pro_dim(self, d):
        """对应 proDim(_:):维度达标且两个已编译模型目录都在才算可用"""
        dim = self.dim(d)
        if dim is None or dim < MIN_DIM or any(n not in self.dirs.get(d, {}) for n in COMPILED):
            return None
        return dim


class Defaults:
    def __init__(self, **kv):
        self.d = kv

    def get(self, k):
        return self.d.get(k)

    def set(self, k, v):
        self.d[k] = v


class World:
    def __init__(self, dirs=None, **kv):
        self.fs = FS(dirs)
        self.defaults = Defaults(**kv)
        self.log = []
        self.notes = []              # 每次启动的 migrationNote
        self.posts = []              # NotificationCenter 已发通知
        self.file_clip_dim = 0       # 库内导入文件的 clip 向量维度(0=没有;对应 fileClipVectorDim)

    def event(self, msg):
        self.log.append(msg)


# ---------------------------------------------------------------- Swift 镜像

def migrate_image_slot(w):
    """migrateImageSlot():三条互斥分支,全部走到末尾清 1.0 记账;只在真正迁移的
    那次启动设 migrationNote(没迁移=None)"""
    fs, d = w.fs, w.defaults
    note = None
    if (dim := fs.pro_dim(PRO)) is not None:            # ① Pro 可用:删 Base 后改名
        if fs.rm(BASE):
            try:
                fs.move(PRO, BASE)
                note = "统一为 so400m"
                w.event("模型槽位迁移(2.0):Pro(dim=%d)已改名为 ImageEmbedder,Base 已删" % dim)
            except OSError:                            # 改名失败:清掉失败现场,不重试
                fs.rm(PRO)
                note = "迁移失败,请重装"
                w.event("模型槽位迁移(2.0):改名失败,已清 Pro 目录")
    elif fs.exists(PRO):                               # ③ Pro 不可用
        fs.rm(PRO)
        if fs.rm(BASE):
            note = "Pro 不可用"
        w.event("模型槽位迁移(2.0):Pro 不可用,已清两个图像目录")
    elif (dim := fs.dim(BASE)) is not None and dim < MIN_DIM:   # ② 只有 Base
        if fs.rm(BASE):
            note = "标准图像模型已停用并移除"
        w.event("模型槽位迁移(2.0):仅 Base(dim=%d),已删" % dim)
    w.notes.append(note)
    d.d.pop("imageModelUsePro", None)


def load_clip(w):
    """loadSlot 的 "clip" 分支。返回 ready / blocked / missing"""
    fs, d = w.fs, w.defaults
    dim = fs.dim(BASE)
    if dim is None:
        return "missing"
    if dim < MIN_DIM:                                   # 2.0 守卫:不用 Base
        return "blocked"
    prev = d.get("clipModelDim")
    d.set("clipModelDim", dim)
    if (prev or 0) != dim:
        w.posts.append("clipModelChanged")
    return "ready"


def import_zip(w, entries):
    """importModelZip():entries = {zip 根目录名: meta 字典}。返回 (导入槽位数, 抛错文案|None)"""
    found, skipped = 0, False
    for name, meta in entries.items():
        if name not in ("TextEmbedder", "ImageEmbedder", "FaceEmbedder"):
            continue                                   # guard ModelSlot(rawValue:)
        if name == BASE and isinstance(meta, dict) and meta.get("dim", MIN_DIM) < MIN_DIM:
            skipped = True                             # 2.0:停用的 Base 不导入
            continue
        w.fs.put(name, meta)
        found += 1
    if found == 0:
        return 0, ("只有已停用的 Base" if skipped else "没有模型目录")
    return found, None


def installed_fingerprint(w, key):
    """installedFingerprint(forPackage:):缺任一槽位返回 None"""
    slots = [BASE] if key == "so400m" else ["TextEmbedder", "FaceEmbedder"]
    parts = []
    for s in slots:
        raw = w.fs.dirs.get(s, {}).get("meta.json")
        if raw is None:
            return None
        m = json.loads(raw)
        if s == "TextEmbedder":
            parts.append("T:%d" % m["dim"])
        elif s == BASE:
            parts.append("P:%d" % m["dim"])
        else:
            parts.append("F:%d:%d" % (m["dim"], m.get("model_size", 0)))
    return "|".join(parts)


def wait_ready(w, text_ready, load_pass_running):
    """waitReady():模型状态在轮询间不变,所以一轮判定即最终结果。
    返回 True=就绪继续同步;等满超时的情况返回 anyReady"""
    image = load_clip(w)
    if text_ready and image == "ready":
        return True
    if not load_pass_running and image == "missing":
        return text_ready          # 2.0:图像模型确实没装→提前返回,不干等满 120 秒
    return text_ready or image == "ready"   # 其余等满超时:anyReady


def sync_imported_files(w, clip_ready, files, broken=()):
    """syncImportedFiles():files = [(文件名, 是否已索引)];broken = 写不出向量的文件名。
    返回本轮处理的文件名列表。视觉文件处理成功后,库内文件 clip 向量维度=当前模型维度"""
    d = w.defaults
    model_dim = d.get("clipModelDim")
    rebuild = False
    if clip_ready and model_dim is not None:
        if (d.get("fileClipRebuiltDim") or 0) != model_dim:
            rebuild = w.file_clip_dim != model_dim
            if not rebuild:
                d.set("fileClipRebuiltDim", model_dim)
    done = []
    for name, indexed in files:
        visual = is_visual(name)
        pick = (clip_ready or not visual) if not indexed else (rebuild and visual)
        if not pick:
            continue
        done.append(name)
        if visual and clip_ready and name not in broken:
            w.file_clip_dim = model_dim
    if rebuild and model_dim is not None:              # 整轮做完才记账
        d.set("fileClipRebuiltDim", model_dim)
    return done


def expect(cond, msg):
    if not cond:
        print("FAIL:", msg)
        sys.exit(1)


# ---------------------------------------------------------------- 场景

def main():
    checks = 0

    def ok(cond, msg):
        nonlocal checks
        expect(cond, msg)
        checks += 1
        print("  ok  " + msg)

    print("S1 全新安装:什么都没装")
    w = World()
    migrate_image_slot(w)
    ok(not w.fs.exists(BASE) and not w.fs.exists(PRO) and w.notes == [None] and not w.log,
       "两个图像目录都不存在→无动作、无提示、无日志")
    ok(load_clip(w) == "missing", "图像模型=未安装")

    print("S2 本人设备(1.0 装了 Base+so400m 且在用 so400m):零重嵌")
    w = World({BASE: full(META_BASE), PRO: full(META_PRO)}, imageModelUsePro=True, clipModelDim=1152)
    migrate_image_slot(w)
    ok(not w.fs.exists(PRO) and w.fs.dim(BASE) == 1152, "Base 删除,Pro 目录改名顶替为 ImageEmbedder")
    ok(w.notes == ["统一为 so400m"] and len(w.log) == 1, "提示迁移完成,记 1 条日志")
    ok("imageModelUsePro" not in w.defaults.d, "1.0 的槽位选择记账已清除")
    ok(load_clip(w) == "ready" and w.posts == [], "维度 1152 未变→不广播→向量库不动(零重嵌)")
    fp_full, fp_pro = installed_fingerprint(w, "full"), installed_fingerprint(w, "so400m")
    w.fs.put("TextEmbedder", META_TEXT); w.fs.put("FaceEmbedder", META_FACE)
    ok(installed_fingerprint(w, "full") == "T:512|F:512:174383860",
       "基础包指纹 == 在线源新清单,且 == 旧指纹去掉 I 段(升级后显示无需重装)")
    ok(installed_fingerprint(w, "so400m") == "P:1152" and fp_full is None and fp_pro == "P:1152",
       "图像模型包指纹 P:1152 == 在线源;文本/人脸没装时基础包指纹为 nil")

    print("S2' 迁移幂等:再启动一次")
    w.log.clear()
    migrate_image_slot(w)
    ok(w.notes[-1] is None and not w.log and w.fs.dim(BASE) == 1152,
       "第二次启动无动作、无新提示(notes 按启动累积,只看最后一次)")

    print("S3 1.0 装了 so400m 但选的是标准(库里 768 维):照常全量重嵌")
    w = World({BASE: full(META_BASE), PRO: full(META_PRO)}, imageModelUsePro=False, clipModelDim=768)
    migrate_image_slot(w)
    ok(w.fs.dim(BASE) == 1152, "同样改名顶替")
    load_clip(w)
    ok(w.posts == ["clipModelChanged"], "维度 768→1152→广播一次(同步开始处清旧向量重嵌)")

    print("S4 只装过 1.0 完整包(只有 Base)")
    w = World({BASE: full(META_BASE)}, clipModelDim=768)
    migrate_image_slot(w)
    ok(not w.fs.exists(BASE) and w.notes == ["标准图像模型已停用并移除"], "Base 停用并删除,提示去下载 so400m")
    ok(load_clip(w) == "missing" and not w.posts, "图像模型=未安装,不会拿 Base 顶上")
    ok(wait_ready(w, text_ready=True, load_pass_running=False) is True,
       "waitReady:加载轮次已结束且图像模型没装→立即返回,不干等满超时")

    print("S4' 随后下载 so400m:触发重嵌")
    found, err = import_zip(w, {BASE: META_PRO})
    ok(found == 1 and err is None and load_clip(w) == "ready", "导入 so400m 后就绪")
    ok(w.posts == ["clipModelChanged"] and w.defaults.get("clipModelDim") == 1152,
       "旧记录 768→1152→广播一次(库里的 Base 向量被清并重嵌)")
    w.posts.clear()
    import_zip(w, {BASE: META_PRO}); load_clip(w)
    ok(w.posts == [], "同模型重装:维度不变→不广播→不重嵌")

    print("S5 Pro 目录损坏")
    for label, content in (("meta.json 缺失", {}), ("meta 不是 JSON", {"meta.json": "{坏"}),
                           ("meta 缺 dim", {"meta.json": json.dumps({"type": "clip"})})):
        w = World({BASE: full(META_BASE), PRO: dict(content)})
        migrate_image_slot(w)
        ok(not w.fs.exists(PRO) and not w.fs.exists(BASE) and w.notes == ["Pro 不可用"],
           "%s→Pro 与 Base 都清掉,提示重下" % label)

    print("S6 迁移中断后续做")
    w = World({PRO: full(META_PRO)})                              # 删了 Base 后被杀
    migrate_image_slot(w)
    ok(w.fs.dim(BASE) == 1152 and not w.fs.exists(PRO), "只剩 Pro→下次启动直接改名")
    w = World({BASE: full(META_PRO)}, imageModelUsePro=True)      # 改名后被杀
    migrate_image_slot(w)
    ok(w.notes == [None] and not w.log and "imageModelUsePro" not in w.defaults.d,
       "已改名→无动作,但仍清掉旧记账")

    print("S7 终态可回滚:ImageEmbedder 里就是 so400m 的 meta")
    w = World({BASE: full(META_BASE), PRO: full(META_PRO)}, clipModelDim=1152)
    migrate_image_slot(w)
    ok(json.loads(w.fs.dirs[BASE]["meta.json"]) == META_PRO and not w.fs.exists(PRO),
       "1.0 会把 ImageEmbedder 当标准槽加载到 1152 维模型,与库内维度一致,不触发重嵌")

    print("S8 手动塞进 1.0 时代的旧包")
    w = World()
    found, err = import_zip(w, {"TextEmbedder": META_TEXT, BASE: META_BASE, "FaceEmbedder": META_FACE})
    ok(found == 2 and err is None and not w.fs.exists(BASE)
       and installed_fingerprint(w, "full") == "T:512|F:512:174383860",
       "旧完整包:文本/人脸照常导入,Base 目录跳过,不整包报错")
    found, err = import_zip(w, {BASE: META_BASE})
    ok(found == 0 and err == "只有已停用的 Base", "只有 Base 的包:报错并指路在线更新")
    found, err = import_zip(w, {"ImageEmbedderPro": META_PRO})
    ok(found == 0 and err == "没有模型目录", "ImageEmbedderPro 已不是合法槽位名")

    print("S9 最终守卫:Base 删不掉也不会被加载")
    w = World({BASE: full(META_BASE)}, clipModelDim=768)
    w.fs.rmtree_fail = {BASE}
    migrate_image_slot(w)
    ok(w.fs.exists(BASE) and w.notes == [None], "删除失败:目录原样不动,下次启动再试")
    ok(load_clip(w) == "blocked" and w.defaults.get("clipModelDim") == 768 and not w.posts,
       "loadSlot 守卫拦下 Base:不就绪、不改维度记录、不广播")
    w.fs.rmtree_fail = set()
    migrate_image_slot(w)
    ok(not w.fs.exists(BASE), "故障消除后下次启动删掉")

    print("S10 首次安装 so400m")
    w = World()
    load_clip(w)
    import_zip(w, {BASE: META_PRO})
    ok(load_clip(w) == "ready" and w.posts == ["clipModelChanged"],
       "维度从未记录→1152 广播一次(与 1.0 setImageModel 判定等价;全新设备的自动同步由首索引门控挡住)")

    print("S11 waitReady 的提前返回不误伤")
    ok(wait_ready(World({BASE: full(META_PRO)}), text_ready=False, load_pass_running=True) is True,
       "文本没就绪但图像就绪:不提前返回,等满超时后 anyReady=True")
    ok(wait_ready(World(), text_ready=False, load_pass_running=True) is False,
       "两个都没装:等满超时后 anyReady=False")
    ok(wait_ready(World(), text_ready=True, load_pass_running=True) is True,
       "加载轮次还在跑(图像模型只是没轮到):不提前返回,值同超时分支")
    ok(wait_ready(World({BASE: full(META_PRO)}), text_ready=True, load_pass_running=False) is True,
       "模型齐全→True")

    print("S12 Pro 是半成品(有 meta.json、没 mlmodelc)")
    w = World({BASE: full(META_BASE), PRO: half(META_PRO)}, imageModelUsePro=True)
    migrate_image_slot(w)
    ok(not w.fs.exists(PRO) and not w.fs.exists(BASE) and w.notes == ["Pro 不可用"],
       "meta 达标但缺已编译模型→不算可用:两个都清,提示重下(不会删掉 Base 后才加载失败)")
    part = full(META_PRO); del part["SiglipText.mlmodelc"]
    ok(World({"x": part}).fs.pro_dim("x") is None and World({"x": full(META_PRO)}).fs.pro_dim("x") == 1152,
       "只缺一个 mlmodelc 也不可用;两个都在才可用")

    print("S13 Pro 改名失败")
    w = World({BASE: full(META_BASE), PRO: full(META_PRO)}, imageModelUsePro=True)
    w.fs.move_fail = True
    migrate_image_slot(w)
    ok(not w.fs.exists(PRO) and not w.fs.exists(BASE) and w.notes == ["迁移失败,请重装"],
       "Base 已删、失败现场 Pro 被清→收敛到「未安装」并提示重装")
    w.fs.move_fail = False
    w.log.clear()
    migrate_image_slot(w)
    ok(w.notes[-1] is None and not w.log, "再启动无动作(不重试改名)")
    import_zip(w, {BASE: META_PRO})                               # 用户按提示重装
    migrate_image_slot(w)
    ok(w.fs.exists(BASE) and load_clip(w) == "ready", "重装进 ImageEmbedder 的 so400m 不会被下次启动删掉")

    # ---- 导入文件的视觉向量(库内文件 clip 向量维度、文件列表) ----
    def fw(file_dim, rebuilt=None):
        w = World({BASE: full(META_PRO)}, clipModelDim=1152, fileClipRebuiltDim=rebuilt)
        w.file_clip_dim = file_dim
        return w
    docs = [("a.pdf", True), ("b.txt", True), ("c.jpg", True), ("d.mp4", True), ("e.pdf", False)]

    print("F1 在用 so400m、库内文件向量已是 1152 维:文件侧零重做")
    w = fw(1152)
    ok(sync_imported_files(w, True, docs) == ["e.pdf"] and w.defaults.get("fileClipRebuiltDim") == 1152,
       "只做未索引的 e.pdf,记账直接对齐")

    print("F2 文件向量被换模型清掉(从 Base 切到 so400m):全部图片/视频重做一次")
    w = fw(0)
    ok(sync_imported_files(w, True, docs) == ["c.jpg", "d.mp4", "e.pdf"] and w.file_clip_dim == 1152,
       "已索引的 c.jpg/d.mp4 也重做;文本/PDF 已索引的不动")
    ok(sync_imported_files(w, True, [("a.pdf", True), ("b.txt", True), ("c.jpg", True), ("d.mp4", True), ("e.pdf", True)]) == [],
       "记账后再同步不重做")

    print("F3 图像模型没装(Base 已停用、so400m 未下载)")
    w = fw(0, 768)
    new = [("c.jpg", False), ("d.mp4", False), ("b.txt", False)]
    ok(sync_imported_files(w, False, new) == ["b.txt"] and w.defaults.get("fileClipRebuiltDim") == 768,
       "只做文本文件;图片/视频不做也不记已索引,记账不动")
    ok(sync_imported_files(w, True, [("c.jpg", False), ("d.mp4", False), ("b.txt", True)])
       == ["c.jpg", "d.mp4"] and w.defaults.get("fileClipRebuiltDim") == 1152,
       "装好 so400m 后图片/视频补上(b.txt 上轮已做完不重做),记账更新")

    print("F4 没有图片/视频文件")
    w = fw(0)
    ok(sync_imported_files(w, True, [("a.pdf", True)]) == [] and w.defaults.get("fileClipRebuiltDim") == 1152,
       "队列为空,直接记账")

    print("F5 个别图片写不出向量:只重做一轮,不循环")
    w = fw(0)
    ok(sync_imported_files(w, True, [("bad.jpg", True), ("c.jpg", True)], broken={"bad.jpg"}) == ["bad.jpg", "c.jpg"],
       "两个都重做一次")
    ok(sync_imported_files(w, True, [("bad.jpg", True), ("c.jpg", True)], broken={"bad.jpg"}) == [],
       "记账后不再重做(不会每次同步都重跑)")

    print("F6 全新安装:文件全部未索引")
    w = fw(0)
    ok(sync_imported_files(w, True, [(n, False) for n, _ in docs]) == [n for n, _ in docs],
       "图片/视频/文本/PDF 全部处理")

    print(f"\nPASS: 2.0 图像模型槽位迁移与导入文件视觉向量重建 {checks} 项状态机断言全部通过")


if __name__ == "__main__":
    main()
