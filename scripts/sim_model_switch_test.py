#!/usr/bin/env python3
"""第四十六批(so400m/标准图像模型切换)逻辑模拟测试。

本机无 Xcode,CI 在美国高峰拒单——把 Swift 改动的控制流**忠实移植**成
Python 状态机,在模拟文件系统/UserDefaults/向量库上跑全部切换场景断言。
⚠️ 这是逻辑镜像不是 Swift 编译验证:语法/类型错仍要等 CI(亚洲时段);
Swift 代码以 ModelManager.swift / SettingsView.swift 为准,改 Swift 必同步改这里。

覆盖的场景(对应真实用户操作):
  1. 全新启动(只有随包标准模型)→ 加载标准,维度 768
  2. 在线更新导入 so400m(走 key 路由 .imagePro)→ 标准槽不被覆盖
  3. 导入后自动切换 → 维度 1152 + 广播 → 自动同步清旧向量全量重嵌
  4. 设置页切回标准 → 维度 768 + 再重嵌(对称)
  5. 切换后没同步就搜 → 语义通道安全跳过(维度守卫),不崩不乱配
  6. 重启后启动自动同步 → 同样触发清空重嵌(不等手动)
  7. so400m 包被删但选择还留着 → activeImageSlot 自动退回标准
  8. 手动 zip 导入(无 key 上下文)dim=1152 → 启发式路由 .imagePro,标准槽安全
  9. 手动导入标准包(dim=768)→ 照旧替换标准槽(原语义)
 10. so400m 加载失败(meta 损坏)→ 自动退回标准,搜索可用
 11. 切换进行中再点切换 → 并发护栏挡住
 12. 已在 so400m 再导一次(重装)→ setImageModel 空转,不触发重嵌风暴
"""
import json
import shutil


# ---------------------------------------------------------------- 模拟基座

class Defaults:                       # UserDefaults.standard
    def __init__(self):
        self.d = {}

    def bool_(self, k):
        return self.d.get(k, False)

    def obj(self, k):
        return self.d.get(k)

    def set(self, v, k):
        self.d[k] = v


class FakeFS:                         # ApplicationSupport/MediaSeek/Models
    def __init__(self):
        self.files = {}               # 相对路径 -> 内容(json dict 表示 meta)

    def exists(self, path):
        return path in self.files

    def write(self, path, content):
        self.files[path] = content

    def rm_tree(self, prefix):
        self.files = {p: c for p, c in self.files.items() if not p.startswith(prefix)}


class Store:                          # VectorStore(只留本测试关心的部分)
    def __init__(self):
        self.clip_dim = 0             # 库内向量维度(0=空)
        self.clip_rows = 0            # 有视觉向量的照片数
        self.dim_guard_skips = 0      # 查询期维度不匹配被安全跳过的次数

    def clip_vector_dim(self):
        return self.clip_dim

    def wipe_clip_vectors(self):
        self.clip_dim = 0
        self.clip_rows = 0

    def embed_all(self, n, dim):
        self.clip_dim = dim
        self.clip_rows = n

    def search(self, query_dim):
        """VectorStore.searchVectors:405 guard dim == q.count else continue"""
        if self.clip_dim != query_dim:
            self.dim_guard_skips += 1
            return []
        return list(range(self.clip_rows))


META_STD = {"type": "clip", "dim": 768, "image_size": 256, "image_mean": 0.5, "image_std": 0.5}
META_PRO = {"type": "clip", "dim": 1152, "image_size": 384, "image_mean": 0.5, "image_std": 0.5}
BUNDLE = {"TextEmbedder/meta.json": {"type": "gemma", "dim": 768},
          "ImageEmbedder/meta.json": META_STD,
          "FaceEmbedder/meta.json": {"type": "face", "dim": 128}}


# ------------------------------------------------- ModelManager(逻辑镜像)

class ModelManager:
    PRO = "ImageEmbedderPro"

    def __init__(self, fs, defaults, store):
        self.fs, self.defaults, self.store = fs, defaults, store
        self.siglip_dim = None        # siglip 实例是否在内存(用维度表示)
        self.image_switching = False
        self.notifications = []       # NotificationCenter 广播记录
        self.refresh_pro_installed()

    # -- 槽位/选择
    def refresh_pro_installed(self):
        self.pro_installed = self.fs.exists(f"{self.PRO}/meta.json")

    @property
    def use_pro_image(self):
        return self.defaults.bool_("imageModelUsePro")

    @property
    def active_image_slot(self):
        return self.PRO if (self.use_pro_image and self.pro_installed) else "ImageEmbedder"

    # -- 加载(对应 loadSlot 的 clip 分支 + 槽位解析)
    def load_slot(self, slot):
        meta = self.fs.files.get(slot + "/meta.json")
        if meta is None:
            # 随包编译兜底只对三个标准槽位有效;imagePro 无随包
            bundled = BUNDLE.get(slot + "/meta.json")
            if bundled is not None:
                self.fs.write(slot + "/meta.json", bundled)
                meta = bundled
            else:
                raise RuntimeError(
                    "so400m 增强包未导入(设置→模型在线更新 下载后可切换)"
                    if slot == self.PRO else f"未安装模型包 {slot}")
        # 镜像 Codable:meta.json 类型不符(dim 非整数)→ 解码抛错
        if (not isinstance(meta, dict) or meta.get("type") != "clip"
                or not isinstance(meta.get("dim"), int)):
            raise RuntimeError("meta 解码/模型加载失败")
        self.siglip_dim = meta["dim"]                      # SigLIPEmbedder 就绪
        self.defaults.set(meta["dim"], "clipModelDim")     # ModelManager.swift:188

    def load_all(self):
        if self.siglip_dim is None:
            self.load_slot(self.active_image_slot)

    # -- 切换(对应 setImageModel,逐句镜像)
    def set_image_model(self, pro: bool):
        if self.image_switching:                # 并发护栏
            return "blocked"
        if pro == self.use_pro_image:           # 空转(重装场景)
            return "noop"
        if pro and not self.pro_installed:
            return "not-installed"
        self.image_switching = True
        try:
            prev_dim = self.defaults.obj("clipModelDim")
            self.defaults.set(pro, "imageModelUsePro")
            self.siglip_dim = None              # 释放旧模型
            try:
                self.load_slot(self.active_image_slot)
            except RuntimeError:
                self.siglip_dim = None
            # so400m 加载失败自动退回标准
            if self.siglip_dim is None and pro:
                self.defaults.set(False, "imageModelUsePro")
                self.load_slot("ImageEmbedder")
            if self.siglip_dim is not None:
                now_dim = self.defaults.obj("clipModelDim")
                if now_dim != prev_dim:
                    self.notifications.append("clipModelChanged")
            return "ok"
        finally:
            self.image_switching = False

    # -- 导入(对应 importModelZip 的路由段)
    def import_zip(self, entries, image_target=None):
        """entries: {目录名: meta_dict};只模拟图像槽路由与落盘"""
        found = 0
        for name, meta in entries.items():
            slot = name
            if slot == "ImageEmbedder":
                if image_target == self.PRO:
                    slot = self.PRO
                elif image_target is None and meta["dim"] >= 1024:
                    slot = self.PRO              # 高维启发式:防手动导入覆盖标准槽
            self.fs.rm_tree(slot + "/")
            for suffix, content in (("", meta),):
                self.fs.write(f"{slot}/meta.json{suffix}", content)
            found += 1
        assert found > 0
        self.refresh_pro_installed()
        self.siglip_dim = None                   # imageState=.loading → loadAll 重载
        self.load_all()


# --------------------------------------- IndexingCoordinator(同步开始段)

def sync_start(mgr: ModelManager, store: Store, photos=500):
    """IndexingCoordinator:200-206 维度比对 → 清空;known 只认 clip 行"""
    want = mgr.defaults.obj("clipModelDim")
    if want is not None and store.clip_vector_dim() > 0 and store.clip_vector_dim() != want:
        store.wipe_clip_vectors()
    known = store.clip_rows
    if known == 0:                               # 全量重嵌
        store.embed_all(photos, want)
    return store.clip_rows


def auto_sync_observer(mgr):                     # MediaSeekApp 的 clipModelChanged sink
    if "clipModelChanged" in mgr.notifications:
        mgr.notifications.remove("clipModelChanged")
        return "startAutoSyncIfAuthorized"
    return None


# ---------------------------------------------------------------- 断言

def expect(cond, msg):
    if not cond:
        raise AssertionError("FAIL: " + msg)
    print(f"  ✓ {msg}")


def main():
    fs, defaults, store = FakeFS(), Defaults(), Store()
    mgr = ModelManager(fs, defaults, store)

    print("场景1 全新启动:只有随包标准模型")
    mgr.load_all()
    expect(mgr.siglip_dim == 768, "加载标准模型,维度 768")
    expect(mgr.active_image_slot == "ImageEmbedder", "激活槽位=标准")
    n = sync_start(mgr, store, photos=500)
    expect(n == 500 and store.clip_dim == 768, "首建索引:500 张照片以 768 维入库")

    print("场景2 在线更新导入 so400m(key=so400m 路由 .imagePro)")
    mgr.import_zip({"ImageEmbedder": META_PRO}, image_target="ImageEmbedderPro")
    expect(mgr.pro_installed, "proInstalled=True")
    expect(fs.exists("ImageEmbedder/meta.json"), "标准槽原封未动(并存)")
    expect(mgr.siglip_dim == 768, "导入后仍加载当前选择(标准),不擅自换模型")

    print("场景3 导入后自动切换到 so400m(ModelUpdateSection.install)")
    r = mgr.set_image_model(True)
    expect(r == "ok" and mgr.siglip_dim == 1152, "切换成功,内存中模型=so400m(1152)")
    expect(mgr.notifications == ["clipModelChanged"], "维度 768→1152 广播 clipModelChanged")
    expect(auto_sync_observer(mgr) == "startAutoSyncIfAuthorized", "App 观察者收到→自动开同步")
    sync_start(mgr, store, photos=500)
    expect(store.clip_dim == 1152 and store.clip_rows == 500, "旧 768 向量被清空,500 张全量重嵌为 1152")

    print("场景4 设置页切回标准")
    r = mgr.set_image_model(False)
    expect(r == "ok" and mgr.siglip_dim == 768, "切回成功(768)")
    expect("clipModelChanged" in mgr.notifications, "1152→768 同样广播")
    auto_sync_observer(mgr)
    sync_start(mgr, store, photos=500)
    expect(store.clip_dim == 768, "再重嵌回 768(对称,无残留状态)")

    print("场景5 切到 so400m 后、同步前来搜索(维度守卫)")
    mgr.set_image_model(True)
    auto_sync_observer(mgr)                      # 广播已收到但同步还没跑完
    hits = store.search(1152)                    # 查询向量=新模型 1152 维
    expect(hits == [] and store.dim_guard_skips >= 1,
           "库内旧维向量被 dim==q.count 守卫安全跳过,不崩不乱配")

    print("场景6 切换后直接杀 App,重启走启动自动同步")
    mgr2 = ModelManager(fs, defaults, store)   # 模拟重启:fs/defaults/store 持久,管理器重建
    mgr2.notifications = []
    mgr2.load_all()
    expect(mgr2.siglip_dim == 1152, "重启后加载 so400m(选择已持久化)")
    sync_start(mgr2, store, photos=500)
    expect(store.clip_dim == 1152, "启动同步的维度比对同样清空重嵌,不依赖手动")

    print("场景7 so400m 目录被删但选择残留")
    fs.rm_tree("ImageEmbedderPro/")
    mgr3 = ModelManager(fs, defaults, store)
    mgr3.load_all()
    expect(mgr3.active_image_slot == "ImageEmbedder" and mgr3.siglip_dim == 768,
           "activeImageSlot 自动退回标准,App 永不失明")

    print("场景8 手动 zip 导入 so400m(无 key 上下文,启发式路由)")
    mgr3.import_zip({"ImageEmbedder": META_PRO})            # image_target=None
    expect(mgr3.pro_installed, "dim=1152≥1024 → 落 .imagePro")
    expect(fs.exists("ImageEmbedder/meta.json"), "标准槽仍安全")

    print("场景9 手动导入标准包(dim=768)→ 替换标准槽(原语义不变)")
    # 场景8 重装 so400m 后,场景6 留下的「选择=pro」自动复活(active=PRO)——
    # 这是可接受行为(在线更新导入本来就会自动切换);先显式切回标准再测替换
    r = mgr3.set_image_model(False)
    expect(r == "ok" and mgr3.siglip_dim == 768, "显式切回标准(顺带验证重装后选择复活路径)")
    auto_sync_observer(mgr3)
    sync_start(mgr3, store, photos=500)
    mgr3.import_zip({"ImageEmbedder": META_STD})
    expect(fs.exists("ImageEmbedder/meta.json"), "标准槽被替换(原行为)")
    expect(mgr3.siglip_dim == 768, "当前选择是标准,加载照常")

    print("场景10 so400m 加载失败(meta 损坏)自动退回标准")
    fs.write("ImageEmbedderPro/meta.json", {"type": "clip", "dim": "损坏"})
    mgr3.refresh_pro_installed()
    r = mgr3.set_image_model(True)               # 内部加载 PRO 失败→兜底段生效
    expect(r == "ok", "切换调用正常返回(失败被兜底吞掉)")
    expect(mgr3.siglip_dim == 768, "坏包→自动退回标准,语义搜索保活")
    expect(mgr3.use_pro_image is False, "选择回写为标准,下次启动不再撞坏包")
    expect(mgr3.notifications == [], "维度没变化成功→无广播,不会触发重嵌风暴")

    print("场景11 切换进行中重复触发(并发护栏)")
    mgr3.image_switching = True
    r = mgr3.set_image_model(True)
    expect(r == "blocked", "imageSwitching 护栏挡住并发双加载")
    mgr3.image_switching = False

    print("场景12 已在 so400m 时重装(空转,无重嵌风暴)")
    fs.write("ImageEmbedderPro/meta.json", META_PRO)
    defaults.set(True, "imageModelUsePro")
    mgr4 = ModelManager(fs, defaults, store)
    mgr4.load_all()
    expect(mgr4.siglip_dim == 1152, "当前已是 so400m")
    r = mgr4.set_image_model(True)
    expect(r == "noop", "setImageModel(pro:true) 空转返回")
    expect(mgr4.notifications == [], "无广播→不会触发清空重嵌(重装同版本安全)")

    print("\nPASS: 12 个切换场景全部通过")


if __name__ == "__main__":
    main()
