# MediaSeek「智搜」— 端侧语义搜索你的照片、视频与文件

用自然语言找东西:"一只猫" → 秒出所有含猫的图片。全部推理在你的 iPhone 本机完成,**不上传任何数据**。

- **快捷导入**:一键授权系统相册(照片+视频全量、增量自动同步);支持从「文件」App 导入任意文件或整个文件夹(安全域书签,不复制文件、不占双倍空间)
- **模型**:
  - `EmbeddingGemma`(308M,多语言文本嵌入)→ 查询、文件名、文档/文本块、照片 Vision 标签
  - `SigLIP`(图文对比模型)→ 图片/视频关键帧的视觉向量,自然语言查图
  - 中文查图走双路:SigLIP 视觉空间 + EmbeddingGemma 空间的照片标签("photo tags: cat, pet")语义匹配
- **检索**:向量存 SQLite(归一化 float32),vDSP 余弦相似度暴力扫描,数万条目毫秒级
- **目标设备**:iPhone(适配 iPhone 17 Pro Max),最低部署 iOS 17

## 目录结构

```
MediaSeek/
├── project.yml                  # XcodeGen 工程定义
├── scripts/export_models.py     # 模型 → Core ML 导出脚本(在电脑上运行)
└── MediaSeek/                   # App 源码(SwiftUI)
    ├── App/                     # 入口与全局依赖
    ├── Core/                    # 向量库/模型/索引/检索/相册/导入
    └── Views/                   # 搜索页/资料库/设置/组件
```

## 构建步骤

### 1. 导出模型(任一台有 Python 的电脑)

```bash
pip install torch "transformers>=4.55" "coremltools>=8.0" sentencepiece
cd MediaSeek
python scripts/export_models.py --outdir MediaSeek/Resources/Models --zip MediaSeekModels.zip
```

- 产物同时放两处用途:`Resources/Models/` 会随 App 打包;`MediaSeekModels.zip` 也可在 App 内「设置 → 导入模型包」导入(二者选一即可)。
- **EmbeddingGemma 2**:脚本默认 `google/embeddinggemma`。若 EmbeddingGemma 2 已发布,`--gemma <HF仓库名>` 即可,导出接口与 App 侧完全一致,无需改 Swift 代码。
- **中文查图质量**:默认 `google/siglip2-base-b16-256` 偏英文;建议用 `--siglip <多语言版本仓库>`(如 SigLIP2 的 multilingual 检查点,以 Hugging Face 实际名称为准)提升中文直接命中的效果。即便不换,中文查询也会通过 EmbeddingGemma 的照片标签通道命中常见物体(猫/狗/海边等)。

### 2. 生成 Xcode 工程(需 Mac + Xcode 16+)

```bash
brew install xcodegen
cd MediaSeek
xcodegen generate
open MediaSeek.xcodeproj
```

### 3. 真机运行

1. Signing & Capabilities 里选你的开发者团队(个人免费 Apple ID 即可真机调试)
2. iPhone 17 Pro Max 用数据线连接,选择设备后 Cmd+R
3. 首次启动:同意相册权限 → 「资料库 → 同步新增内容」开始建索引;或在「设置」导入模型 zip

## 使用

1. **搜索页**:输入"一只猫""海边的日落""会议纪要""发票 PDF"等自然语言,范围可选 全部/照片/视频/文件,结果按语义相似度排序,点击进入照片查看/视频播放/文件 QuickLook
2. **资料库页**:授权相册、查看统计、同步/重建索引、从「文件」导入文件或文件夹、管理已导入文件
3. **设置页**:模型状态与导入、每个视频抽帧数、清空索引

## 设计说明与已知限制

- **双向量空间**:SigLIP 空间(图+图查文本)与 EmbeddingGemma 空间(查询↔文档/标签)不互通,检索时两路分别召回再按相似度合并去重(同一张照片可能被两条通路同时命中,取高分)。
- **EmbeddingGemma prompt**:查询用 `task: search result | query: …`,文档用 `title: none | text: …`(官方推荐模板),在 `GemmaTextEmbedder` 中实现。
- **视频**:默认每个视频抽 3 帧(设置里 1–5 可调),按帧向量检索后按视频聚合。
- **文件**:图片走 SigLIP,视频抽帧走 SigLIP,PDF/文本/代码走 EmbeddingGemma 分块嵌入;其余二进制文件按文件名嵌入。docx/xlsx 等富文档暂不解析正文(可自行在 `TextExtractor` 加解码)。
- **导入文件是书签引用**:源文件被移动/删除后无法再访问(索引里仍会命中,详情页打不开时会有提示)。
- **性能参考**(iPhone 17 Pro Max,ANE):照片索引约 20–40 张/秒,1 万张照片约 5–10 分钟;检索 <10ms。索引首次会全量,之后增量(启动时自动同步新增)。
- **检索是线性扫描**:约 10 万条目内体验流畅;更大规模可平滑升级 HNSW(替换 `VectorStore.search`)。
- **swift-transformers 版本**:若 `AutoTokenizer.from(modelFolder:)` API 在新版有变动,按其 README 调整 `HFTokenizer` 一行即可。

## 常见问题

- **模型状态一直"未安装"?** 确认导出目录包含 `TextEmbedder/`、`ImageEmbedder/`,且每个目录里有 `.mlpackage` + `tokenizer.json` + `meta.json`。
- **搜索结果为空?** 先到「资料库」跑一次同步;索引完成后重试。
- **真机安装失败?** 检查签名团队;模型较大(约 0.5–1GB),确保设备空间充足。
