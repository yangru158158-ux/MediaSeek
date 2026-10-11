import SwiftUI
import Photos
import UniformTypeIdentifiers

struct SettingsView: View {
    @EnvironmentObject var app: AppModel

    @State private var showModelPicker = false
    @State private var importing = false
    @State private var confirmWipe = false
    @State private var videoDiagRunning = false
    @State private var videoDiagText: String?

    var body: some View {
        NavigationStack {
            List {
                modelSection
                tuningSection
                videoDiagSection
                aboutSection
            }
            .navigationTitle("设置")
            .fileImporter(isPresented: $showModelPicker,
                          allowedContentTypes: [.zip],
                          allowsMultipleSelection: false) { result in
                switch result {
                case .success(let urls):
                    guard let url = urls.first else { return }
                    Task {
                        importing = true
                        do {
                            try await app.models.importModelZip(at: url)
                        } catch {
                            app.fail(error)
                        }
                        importing = false
                    }
                case .failure(let error):
                    app.fail(error)
                }
            }
            .confirmationDialog("确定删除全部索引数据?导入文件记录与模型不受影响。",
                                isPresented: $confirmWipe, titleVisibility: .visible) {
                Button("清空索引", role: .destructive) {
                    try? app.store.deleteAll()
                    // 视频处理记账一并清掉,否则黑屏视频/抽帧数快照会挡住重建后的重索引
                    UserDefaults.standard.removeObject(forKey: "vidFrameCfg")
                    UserDefaults.standard.removeObject(forKey: "vidAllBlack")
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
        case .missing: return "未安装"
        case .loading: return "加载中…(首次编译模型需要一些时间)"
        case .ready: return "已就绪"
        case .failed(let m): return "失败:\(m)"
        }
    }

    private var modelSection: some View {
        Section {
            modelRow(.text)
            modelRow(.image)
            if !app.models.detailText.isEmpty {
                Text(app.models.detailText).font(.caption2).foregroundStyle(.secondary)
            }
            Button {
                showModelPicker = true
            } label: {
                if importing {
                    HStack { ProgressView(); Text("导入中…") }
                } else {
                    Label("从「文件」导入模型包(zip)", systemImage: "square.and.arrow.down")
                }
            }
            .disabled(importing)
            Text("模型包由仓库中的 scripts/export_models.py 在电脑上导出,包含 TextEmbedder 与 ImageEmbedder 两个目录。")
                .font(.caption).foregroundStyle(.secondary)
        } header: {
            Text("端侧模型")
        } footer: {
            Text("所有推理均在设备本机完成,照片、视频与文件内容不会上传到任何服务器。")
        }
    }

    // MARK: - 调优

    /// 视频索引诊断:相册视频总数 vs 已有帧向量视频数,
    /// 并列出最旧/最新各 3 个视频的索引状态——定位「视频搜不到」断在哪一环
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

    private var aboutSection: some View {
        Section("关于") {
            LabeledContent("版本", value: "1.0")
            LabeledContent("目标设备", value: "iPhone(iPhone 17 Pro Max 适配)")
            LabeledContent("向量检索", value: "vDSP 余弦相似度")
        }
    }
}
