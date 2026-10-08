import SwiftUI
import UniformTypeIdentifiers

struct SettingsView: View {
    @EnvironmentObject var app: AppModel

    @State private var showModelPicker = false
    @State private var importing = false
    @State private var confirmWipe = false

    var body: some View {
        NavigationStack {
            List {
                modelSection
                tuningSection
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

    private var tuningSection: some View {
        Section("索引选项") {
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
