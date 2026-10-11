import SwiftUI
import Photos

/// 查重入口(搜索页/资料库共用):模式选择→扫描进度→结果弹层
struct DuplicateScanView: View {
    @EnvironmentObject var app: AppModel
    @Environment(\.dismiss) private var dismiss
    @State private var mode = "similar"
    @State private var scanning = false
    @State private var progress: (done: Int, total: Int)?
    @State private var groups: [[String]] = []
    @State private var showResults = false
    @State private var photoCount = 0

    var body: some View {
        NavigationStack {
            Form {
                Section("查重范围") {
                    Picker("模式", selection: $mode) {
                        Text("完全重复").tag("exact")
                        Text("雷同(相似画面)").tag("similar")
                    }
                    .pickerStyle(.segmented)
                    Text("覆盖已索引的照片(\(photoCount) 张);删除走系统相册「最近删除」,30 天内可恢复。")
                        .font(.caption2).foregroundStyle(.secondary)
                }
                Section {
                    Button {
                        start()
                    } label: {
                        if scanning {
                            HStack { ProgressView(); Text("扫描中…") }
                        } else {
                            Label(photoCount < 2 ? "照片不足,先索引照片" : "开始查重",
                                  systemImage: "square.stack.3d.down.right")
                        }
                    }
                    .disabled(scanning || photoCount < 2)
                    if scanning, let p = progress, p.total > 0 {
                        ProgressView(value: Double(p.done) / Double(p.total))
                        Text("已处理 \(p.done)/\(p.total) 张")
                            .font(.caption2).foregroundStyle(.secondary)
                    }
                }
            }
            .navigationTitle("照片查重")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button(scanning ? "后台继续" : "关闭") { dismiss() }
                }
            }
            .sheet(isPresented: $showResults) {
                DuplicateResultsView(initialGroups: groups)
            }
            .onAppear { photoCount = app.indexing.indexedPhotoCount }
        }
    }

    private func start() {
        guard !scanning else { return }
        scanning = true
        progress = (0, 1)
        let threshold: Float = mode == "exact" ? 0.985 : 0.92
        let store = app.store
        Task.detached(priority: .userInitiated) {
            do {
                let data = try store.allPhotoVectors()
                let groups = DuplicateScanner.scan(
                    keys: data.keys, flat: data.flat, dim: data.dim, threshold: threshold) { done, total in
                    Task { @MainActor in progress = (done, total) }
                }
                let keyGroups = groups.map { $0.map { data.keys[$0] } }.filter { $0.count > 1 }
                await MainActor.run {
                    scanning = false
                    progress = nil
                    if keyGroups.isEmpty {
                        app.notify("没有发现重复/雷同照片")
                    } else {
                        self.groups = keyGroups
                        showResults = true
                    }
                }
            } catch {
                await MainActor.run { scanning = false; app.fail(error) }
            }
        }
    }
}

/// 查重结果:分组展示雷同照片,默认每组保留第一张、其余勾选,
/// 批量删除走系统相册「最近删除」(30 天可恢复),同时清除索引。
struct DuplicateResultsView: View {
    let initialGroups: [[String]]
    @EnvironmentObject var app: AppModel
    @Environment(\.dismiss) private var dismiss
    @State private var groups: [[String]] = []
    @State private var assets: [String: PHAsset] = [:]
    @State private var selected: Set<String> = []
    @State private var deleting = false
    @State private var confirmDelete = false

    var body: some View {
        NavigationStack {
            Group {
                if groups.isEmpty {
                    ContentUnavailableView("没有可删除的重复项", systemImage: "checkmark.seal")
                } else {
                    List {
                        ForEach(groups.indices, id: \.self) { gi in
                            groupSection(gi)
                        }
                    }
                }
            }
            .navigationTitle("查重结果")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("关闭") { dismiss() }
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button(selected.isEmpty ? "勾选待删" : "删除所选(\(selected.count))") {
                        confirmDelete = true
                    }
                    .disabled(selected.isEmpty || deleting)
                }
            }
            .confirmationDialog("把所选 \(selected.count) 张移入「最近删除」?",
                                isPresented: $confirmDelete, titleVisibility: .visible) {
                Button("删除(30 天内可恢复)", role: .destructive) { performDelete() }
                Button("取消", role: .cancel) {}
            } message: {
                Text("删除会同时清除其索引与标签;原片在最近删除保留 30 天")
            }
            .onAppear {
                guard groups.isEmpty else { return }
                groups = initialGroups
                for g in groups { selected.formUnion(g.dropFirst()) }   // 默认每组保留第 1 张
                let ids = Array(Set(groups.flatMap { $0 }))
                let fetch = PHAsset.fetchAssets(withLocalIdentifiers: ids, options: nil)
                fetch.enumerateObjects { a, _, _ in assets[a.localIdentifier] = a }
                // 已不在相册里的(被手动删过)从分组剔除
                groups = groups.map { g in g.filter { assets[$0] != nil } }.filter { $0.count > 1 }
                selected = selected.filter { assets[$0] != nil }
            }
        }
    }

    private func groupSection(_ gi: Int) -> some View {
        let g = groups[gi]
        let picked = g.filter { selected.contains($0) }.count
        return Section("第 \(gi + 1) 组 · \(g.count) 张 · 已选删 \(picked) 张") {
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(g, id: \.self) { key in
                        Group {
                            if let asset = assets[key] {
                                AssetThumbView(asset: asset, pixel: 160)
                                    .frame(width: 96, height: 96)
                                    .clipped()
                            } else {
                                Rectangle().fill(Color(.systemGray5))
                                    .frame(width: 96, height: 96)
                            }
                        }
                        .overlay(alignment: .topTrailing) {
                            Image(systemName: selected.contains(key) ? "checkmark.circle.fill" : "circle")
                                .font(.title3)
                                .foregroundStyle(selected.contains(key) ? .blue : .white)
                                .shadow(radius: 2)
                                .padding(4)
                        }
                        .onTapGesture { toggle(key) }
                    }
                }
                .padding(.vertical, 4)
            }
            Button("保留第 1 张,选中本组其余") {
                for key in g.dropFirst() { selected.insert(key) }
            }
            .font(.footnote)
        }
    }

    private func toggle(_ key: String) {
        if selected.contains(key) { selected.remove(key) } else { selected.insert(key) }
    }

    private func performDelete() {
        deleting = true
        let ids = Array(selected)
        Task {
            let fetch = PHAsset.fetchAssets(withLocalIdentifiers: ids, options: nil)
            do {
                try await PHPhotoLibrary.shared().performChanges {
                    PHAssetChangeRequest.deleteAssets(fetch as NSFastEnumeration)
                }
                for id in ids { try? app.store.removeEverythingForRef(id) }
                await MainActor.run {
                    groups = groups.map { g in g.filter { !ids.contains($0) } }.filter { $0.count > 1 }
                    selected.removeAll()
                    deleting = false
                    app.notify("已删除 \(ids.count) 张,最近删除保留 30 天")
                    if groups.isEmpty { dismiss() }
                }
            } catch {
                await MainActor.run { deleting = false; app.fail(error) }
            }
        }
    }
}
