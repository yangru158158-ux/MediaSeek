import SwiftUI
import Photos

struct SettingsView: View {
    @EnvironmentObject var app: AppModel

    @State private var confirmWipe = false
    @State private var confirmRespeech = false
    @State private var videoDiagRunning = false
    @State private var videoDiagText: String?
    @State private var labelVocab: [(label: String, count: Int)] = []
    @State private var showLabelVocab = false
    // 图像模型切换(第四十六批):picker 本地镜像 + 待确认的目标值
    @State private var imageModelPick = false
    @State private var pendingImagePro: Bool?

    var body: some View {
        NavigationStack {
            List {
                modelSection
                imageModelSection
                ModelUpdateSection()
                tuningSection
                labelVocabSection
                videoDiagSection
                aboutSection
            }
            .navigationTitle("设置")
            .sheet(isPresented: $showLabelVocab) {
                LabelVocabSheet(items: labelVocab)
            }
            // 第八十四批:「从文件导入模型包」入口移除——模型安装唯一通道
            // 收敛到「模型在线更新」(断点续传+自动导入),手动 zip 导入只剩
            // 调试价值;importModelZip 函数保留,在线更新下载后走同一函数
            .confirmationDialog("确定删除全部索引数据?导入文件记录与模型不受影响。",
                                isPresented: $confirmWipe, titleVisibility: .visible) {
                Button("清空索引", role: .destructive) {
                    try? app.store.deleteAll()
                    // 视频处理记账一并清掉,否则黑屏视频/抽帧数快照会挡住重建后的重索引
                    UserDefaults.standard.removeObject(forKey: "vidFrameCfg")
                    UserDefaults.standard.removeObject(forKey: "vidAllBlack")
                    UserDefaults.standard.removeObject(forKey: "vidAssetFail")
                    UserDefaults.standard.removeObject(forKey: "vidAssetFailCnt")
                    UserDefaults.standard.removeObject(forKey: "vidNoLabel")
                    // 语音转写记账:库清了转写文本也没了,记账必须一起清否则永不重转
                    UserDefaults.standard.removeObject(forKey: "speechDone")
                    UserDefaults.standard.removeObject(forKey: "speechFail")
                    UserDefaults.standard.removeObject(forKey: "speechFailCnt")
                }
                Button("取消", role: .cancel) {}
            }
        }
    }

    // MARK: - 模型

    @ViewBuilder
    private func modelRow(_ slot: ModelSlot) -> some View {
        let state = app.models.state(for: slot)
        HStack {
            VStack(alignment: .leading, spacing: 3) {
                Text(slot.displayName).font(.subheadline)
                Text(stateText(state))
                    .font(.caption)
                    .foregroundStyle(state == .ready ? .green : .secondary)
            }
            Spacer()
            switch state {
            case .loading:
                ProgressView()
            case .ready:
                Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
            default:
                Image(systemName: "exclamationmark.circle").foregroundStyle(.orange)
            }
        }
    }

    private func stateText(_ state: ModelState) -> String {
        switch state {
        case .missing:
            // 冷启动排队中 vs 真没装:loadAll 进行中的 missing 是「还没轮到」,
            // 显示「未安装」会让用户以为模型被删(2026-10-16 实测)
            return app.models.loadPassRunning ? "加载中(排队)…" : "未安装"
        case .loading: return "加载中…(首次编译模型需要一些时间)"
        case .ready: return "已就绪"
        case .failed(let m): return "失败:\(m)"
        }
    }

    private var modelSection: some View {
        Section {
            modelRow(.text)
            // 图像行显示「当前选中」的图像模型(标准/so400m,第四十六批)
            modelRow(app.models.activeImageSlot)
            modelRow(.face)
            if !app.models.detailText.isEmpty {
                Text(app.models.detailText).font(.caption2).foregroundStyle(.secondary)
            }
        } header: {
            Text("端侧模型")
        } footer: {
            Text("所有推理均在设备本机完成,照片、视频与文件内容不会上传到任何服务器。")
        }
    }

    // MARK: - 图像模型切换(第四十六批:so400m ↔ 标准,A/B 对比用)

    private var imageModelSection: some View {
        Section {
            Picker("当前图像模型", selection: $imageModelPick) {
                Text("标准 SigLIP2(随包)").tag(false)
                Text("高精度 so400m(增强)").tag(true)
            }
            .pickerStyle(.inline)
            .disabled(!app.models.proInstalled || app.models.imageSwitching)
            .onChange(of: imageModelPick) { pick in
                if pick != app.models.useProImage { pendingImagePro = pick }
            }
            .onChange(of: app.models.useProImage) { pick in
                imageModelPick = pick   // 在线更新导入 so400m 自动切换时同步 picker
            }
            .onAppear { imageModelPick = app.models.useProImage }

            if !app.models.proInstalled {
                Label("so400m 未导入:在下方「模型在线更新」下载增强图像模型后即可切换", systemImage: "info.circle")
                    .font(.caption).foregroundStyle(.secondary)
            }
            if app.models.imageSwitching {
                HStack { ProgressView(); Text("正在切换(加载模型)…") }
                    .font(.caption).foregroundStyle(.secondary)
            }
        } header: {
            Text("图像模型切换(实验)")
        } footer: {
            Text(proInstalledFooter)
        }
        .confirmationDialog(
            "切换图像模型?",
            isPresented: Binding(
                get: { pendingImagePro != nil },
                set: { if !$0 { pendingImagePro = nil; imageModelPick = app.models.useProImage } }),
            titleVisibility: .visible
        ) {
            Button(pendingImagePro == true ? "切换到 so400m" : "切换回标准模型") {
                guard let target = pendingImagePro else { return }
                pendingImagePro = nil
                Task { await app.models.setImageModel(pro: target) }
            }
            Button("取消", role: .cancel) {
                pendingImagePro = nil
                imageModelPick = app.models.useProImage
            }
        } message: {
            Text("将自动清空视觉向量并开始全量重嵌(约 1-2 小时,可中断续跑;人脸、文字、标签通道不受影响)。重嵌完成前语义搜索结果为空属正常。")
        }
    }

    private var proInstalledFooter: String {
        app.models.proInstalled
            ? "切换 = 换视觉「眼睛」:两个模型嵌入维度不同,旧视觉向量自动作废清空,切回时同样要重嵌。适合 A/B 对比同一批搜索词(如 woman/男人)的效果。人脸、文字、标签数据不受影响。"
            : "先在下方「模型在线更新」下载 so400m 增强包(约 2GB,建议 Wi-Fi)后,这里就能切换。"
    }

    // MARK: - 调优

    /// 视频索引诊断:相册视频总数 vs 已有帧向量视频数,
    /// 并列出最旧/最新各 3 个视频的索引状态——定位「视频搜不到」断在哪一环
    /// 标签词典:库内照片/视频帧实际打上的全部 Vision 英文标签+次数
    /// ——「图像模型能识别哪些英文词」的精确答案(可被标签通道精确命中的集合)
    private var labelVocabSection: some View {
        Section {
            Button {
                labelVocab = app.search.labelVocabCounts()
                showLabelVocab = true
            } label: {
                HStack {
                    Text("标签词典")
                    Spacer()
                    Text(labelVocab.isEmpty ? "点按载入" : "\(labelVocab.count) 个标签")
                        .foregroundColor(.secondary)
                }
            }
        } header: {
            Text("图像模型词汇")
        } footer: {
            Text("列出库里照片/视频实际打上的英文标签及中文对照。这些中文直接可搜且双通道兼得:文字维度=照片里含该字样,语义/标签维度=画面内容匹配(如搜「海滩」命中 beach 标签的照片)。未收录中文的标签仍可搜英文原词。")
        }
    }

    private var videoDiagSection: some View {
        Section("视频索引诊断") {
            Button {
                runVideoDiag()
            } label: {
                if videoDiagRunning {
                    HStack { ProgressView(); Text("诊断中…") }
                } else {
                    Label("运行诊断", systemImage: "stethoscope")
                }
            }
            .disabled(videoDiagRunning)
            if let t = videoDiagText {
                Text(t).font(.caption2).monospaced().foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
        }
    }

    private func runVideoDiag() {
        videoDiagRunning = true
        let store = app.store
        Task.detached(priority: .userInitiated) {
            var lines: [String] = []
            do {
                let stats = try store.videoFrameStats()
                lines.append("已有帧向量视频数:\(stats.videos)")
                // 帧标签落库情况:判断「标签没写进库」还是「写进了但词对不上」
                let labelRows = (try? store.allVideoFrameLabels()) ?? []
                let labelByRef = Dictionary(labelRows.map { ($0.refKey, $0.title) },
                                            uniquingKeysWith: { a, _ in a })
                lines.append("已打标签视频数:\(Set(labelRows.map { $0.refKey }).count)")
                let opts = PHFetchOptions()
                opts.predicate = NSPredicate(format: "mediaType = %d", PHAssetMediaType.video.rawValue)
                let fetch = PHAsset.fetchAssets(with: .video, options: opts)
                lines.append("相册视频总数:\(fetch.count)")
                var assets: [PHAsset] = []
                fetch.enumerateObjects { a, _, _ in assets.append(a) }
                let sorted = assets.sorted { ($0.creationDate ?? .distantPast) > ($1.creationDate ?? .distantPast) }
                func line(_ a: PHAsset) -> String {
                    let d = a.creationDate.map { DateFormatter.localizedString(from: $0, dateStyle: .medium, timeStyle: .none) } ?? "无日期"
                    let n = stats.perVideo[a.localIdentifier] ?? 0
                    let lab = labelByRef[a.localIdentifier].map { " · 标签:" + String($0.prefix(46)) } ?? " · 无标签"
                    return "  \(d) · 帧\(n > 0 ? "\(n) ✓" : "0 ✗未索引")\(lab)"
                }
                lines.append("最新 3 个:")
                lines.append(contentsOf: sorted.prefix(3).map(line))
                lines.append("最旧 3 个:")
                lines.append(contentsOf: sorted.suffix(3).map(line))
                let missing = assets.filter { (stats.perVideo[$0.localIdentifier] ?? 0) == 0 }
                lines.append("未索引视频:\(missing.count) 个")
                // 读取失败黑名单(iCloud 未下载/损坏,连续失败 3 次跳过;全量重建可重试)
                let failed = UserDefaults.standard.stringArray(forKey: "vidAssetFail") ?? []
                if !failed.isEmpty {
                    lines.append("读取失败已跳过:\(failed.count) 个(未下载/损坏;下载到本机后「全量重建索引」可恢复)")
                }
                if let oldest = missing.map({ ($0.creationDate ?? .distantPast, $0) }).min(by: { $0.0 < $1.0 }) {
                    let d = DateFormatter.localizedString(from: oldest.0, dateStyle: .medium, timeStyle: .none)
                    lines.append("最旧未索引:\(d)")
                }
            } catch {
                lines.append("诊断失败:\(error.localizedDescription)")
            }
                // 语义探针 v2:已知强查询做对照,二分「文本塔整体退化」vs「罕见地名嵌入弱」
                if let siglip = await MainActor.run(body: { app.models.siglip }) {
                    let probes = [("电脑屏幕(中文)", "电脑屏幕"),
                                  ("computer screen(模板)", "This is a photo of a computer screen."),
                                  ("beach(英文)", "beach"),
                                  ("maldives(模板)", "This is a photo of maldives tropical island beach."),
                                  ("马尔代夫(中文)", "马尔代夫")]
                    for probe in probes {
                        if let q = try? siglip.embedQuery(probe.1) {
                            let ph = (try? store.search(space: siglip.space, kinds: [.photo],
                                                        query: q, limit: 1, minScore: -1)) ?? []
                            let vf = (try? store.search(space: siglip.space, kinds: [.videoFrame],
                                                        query: q, limit: 1, minScore: -1)) ?? []
                            let p = ph.first.map { String(format: "%.3f", $0.score) } ?? "无"
                            let f = vf.first.map { String(format: "%.3f", $0.score) } ?? "无"
                            lines.append("\(probe.0):照片 \(p) / 视频帧 \(f)")
                        }
                    }
                    lines.append("(电脑屏幕也 <0.15 → 文本塔整体异常;仅马尔代夫低 → 罕见地名嵌入弱)")
                }
            let out = lines.joined(separator: "\n")
            await MainActor.run {
                videoDiagText = out
                videoDiagRunning = false
            }
        }
        _ = store
    }

    /// 重新转写:清记账+清旧语音文本(含白卷/占位),下轮同步全量重跑
    /// (第三十七批:听写启用后旧白卷不会自愈——speechDone 记账挡着重跑)
    private func resetSpeech() {
        UserDefaults.standard.removeObject(forKey: "speechDone")
        UserDefaults.standard.removeObject(forKey: "speechFail")
        UserDefaults.standard.removeObject(forKey: "speechFailCnt")
        try? app.store.clearSpeechTexts()
        app.notify("已重置转写记录。回资料库点「同步新增内容」开始重跑(可中断可续)")
    }

    /// 语音转写开关下方的状态说明:权限被拒/机型不支持时给出行动指引,
    /// 正常态讲清代价与检索入口(文字:)
    private var speechHint: String {
        if UserDefaults.standard.bool(forKey: "speechDenied") {
            return "语音识别权限被拒:到 系统设置→智搜 打开「语音识别」后,下次同步自动生效"
        }
        if UserDefaults.standard.bool(forKey: "speechUnsupported") {
            return "本机不支持中文端侧语音识别,该功能已停用"
        }
        return "开启后点「同步新增内容」,把视频里说的话转成本机文字(≈音频时长,耗电,建议插电;可中断续跑)。转写文本可被「文字:」搜索;关闭开关不影响已转写内容"
    }

    private var tuningSection: some View {
        Section("索引选项") {
            Toggle(isOn: Binding(
                get: { app.indexing.chineseLabels },
                set: { app.indexing.chineseLabels = $0 })) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("中文语义标签")
                    Text("增强中文物体查询,但索引速度慢 2-3 倍")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            // 第七十批:索引速度——精准(默认,OCR 2048px,远景小字优先) /
            // 极速(1280px,像素量约降 2.5 倍,截图证件大字几乎无损)。
            // 对已索引内容需「全量重建索引」才按新分辨率重做
            Toggle(isOn: Binding(
                get: { !app.indexing.ocrFastMode },
                set: { app.indexing.ocrFastMode = !$0 })) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("精准 OCR(文字识别优先)")
                    Text(app.indexing.ocrFastMode
                         ? "极速:1280px,大字几乎无损,远景小字可能漏,人脸检测分辨率同步降低。对新照片即时生效;已索引的要「全量重建索引」才重做"
                         : "默认:2048px,拍屏角度/小字也尽量认出。对新照片即时生效;已索引的要「全量重建索引」才重做")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            // 批80二修:索引并发数可调(用户找最佳值)。变更对下一次开始/
            // 续跑的索引生效;进行中的一轮不变
            Picker("索引并发数", selection: Binding(
                get: { app.indexing.indexConcurrency },
                set: { UserDefaults.standard.set($0, forKey: "indexConcurrency") })) {
                ForEach(2...8, id: \.self) { Text("\($0) 路").tag($0) }
            }
            // 批91:措辞改「当前 N 路」——「默认 4」曾被读成当前值(用户实测
            // 误会);当前值在行右侧,提示行只说明出厂默认与生效方式
            Text("当前 \(app.indexing.indexConcurrency) 路 · 出厂默认 4;越大越快、越热。改动立即生效(进行中的索引热切换)")
            HStack {
                ShareLink(item: { () -> URL in
                    IndexingCoordinator.DiagLog.shared.ensureFileExists()
                    return FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
                        .appendingPathComponent("diag.log")
                }()) {
                    Label("导出诊断日志", systemImage: "doc.text")
                }
                .font(.caption)
            }
                .font(.caption).foregroundStyle(.secondary)
            Toggle(isOn: Binding(
                get: { UserDefaults.standard.bool(forKey: "speechEnabled") },
                set: { UserDefaults.standard.set($0, forKey: "speechEnabled") })) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("视频语音转文字")
                    Text(speechHint)
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            // 语音识别引擎诊断+服务器回退开关(第三十八批):系统听写和 Speech
            // 框架的本机模型是两套——听写出字不代表 Speech 本机模型在位
            if UserDefaults.standard.bool(forKey: "speechEnabled") {
                VStack(alignment: .leading, spacing: 4) {
                    Text(SpeechTranscriber.diagnosticInfo())
                        .font(.caption).foregroundStyle(.secondary)
                    Toggle(isOn: Binding(
                        get: { UserDefaults.standard.object(forKey: "speechAllowServer") as? Bool ?? true },
                        set: { UserDefaults.standard.set($0, forKey: "speechAllowServer") })) {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("允许服务器识别")
                            Text("本机模型不可用时,音频经苹果服务器处理(与系统听写行为一致)。关闭则强制本机,需确保模型已下载")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                    }
                    if !(UserDefaults.standard.object(forKey: "speechAllowServer") as? Bool ?? true) {
                        Text("💡 下载本机模型:设置→Siri与搜索→语言→改为中文(普通话-中国大陆),等待下载完成后重新转写")
                            .font(.caption2).foregroundStyle(.orange)
                    }
                }
            }
            // 语音转写诊断(第三十七批):听写未启用时识别器全交白卷却记账
            // "已完成",靠这组数暴露真伪
            let diag = app.store.speechDiagnostics()
            let doneN = (UserDefaults.standard.stringArray(forKey: "speechDone") ?? []).count
            VStack(alignment: .leading, spacing: 2) {
                Text("转写状态:已记账 \(doneN) 部 · 已存文本 \(diag.stored) 部")
                    .font(.caption).foregroundStyle(
                        (UserDefaults.standard.bool(forKey: "speechEnabled") && diag.stored == 0 && doneN > 0)
                            ? .red : .secondary)
                if diag.stored > 0, !diag.sample.isEmpty {
                    Text("样例:\(diag.sample)…")
                        .font(.caption2).foregroundStyle(.secondary).lineLimit(1)
                }
                if UserDefaults.standard.bool(forKey: "speechEnabled"), diag.stored == 0, doneN > 0 {
                    Text("已记账但零文本——多为听写未启用导致识别器空转。去 系统设置→通用→键盘→启用听写(勾普通话),然后点下方「重新转写」")
                        .font(.caption2).foregroundStyle(.red)
                }
            }
            Button(role: .destructive) {
                confirmRespeech = true
            } label: {
                Label("重新转写全部视频", systemImage: "arrow.clockwise")
            }
            .confirmationDialog("重新转写会清空全部已转写记录并重跑(≈音频总时长,建议插电)。继续?",
                                isPresented: $confirmRespeech, titleVisibility: .visible) {
                Button("重新转写", role: .destructive) { resetSpeech() }
                Button("取消", role: .cancel) {}
            }
            Stepper(value: Binding(
                get: { app.indexing.videoFramesPerVideo },
                set: { app.indexing.videoFramesPerVideo = $0 }), in: 1...5) {
                HStack {
                    Text("每个视频抽取帧数")
                    Spacer()
                    Text("\(app.indexing.videoFramesPerVideo)")
                        .foregroundStyle(.secondary)
                }
            }
            Button(role: .destructive) {
                confirmWipe = true
            } label: {
                Label("清空索引数据", systemImage: "trash")
            }
        }
    }

    // MARK: - 关于

    /// 第八十四批:版本号此前硬编码 "1.0" 只显示一半——构建号看不到,
    /// 排查问题时对不上构建;改读 Bundle(短版本+构建号,与 TestFlight
    /// 页一致,报问题截图即可定位构建)
    private var appVersionText: String {
        let v = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "1.0"
        let b = Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "?"
        return "\(v)(\(b))"
    }

    private var aboutSection: some View {
        Section("关于") {
            LabeledContent("版本", value: appVersionText)
            LabeledContent("目标设备", value: "iPhone(iPhone 17 Pro Max 适配)")
            LabeledContent("向量检索", value: "vDSP 余弦相似度")
        }
    }
}

/// 标签词典列表:英文标签+对应中文,可按任一语言筛选。
/// 展示的中文本身可搜:文字维度=照片含该字样(OCR),语义/标签维度=词典反查命中英文标签
struct LabelVocabSheet: View {
    let items: [(label: String, count: Int)]
    @State private var filter = ""
    @Environment(\.dismiss) private var dismiss

    private var filtered: [(label: String, count: Int)] {
        filter.isEmpty ? items : items.filter {
            $0.label.localizedCaseInsensitiveContains(filter)
                || (LabelZh.chinese(for: $0.label) ?? "").contains(filter)
        }
    }

    var body: some View {
        NavigationStack {
            List(filtered, id: \.label) { item in
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(LabelZh.chinese(for: item.label) ?? "—")
                            .font(.subheadline)
                        Text(item.label).font(.caption.monospaced())
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    Text("\(item.count)").font(.footnote).foregroundColor(.secondary)
                }
            }
            .searchable(text: $filter, prompt: "筛选,如 海滩 / beach / 食物")
            .navigationTitle("标签词典 \(items.count) 项")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("完成") { dismiss() }
                }
            }
        }
    }
}
