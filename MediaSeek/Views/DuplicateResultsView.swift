import SwiftUI
import Photos

/// 查重入口(搜索页/资料库共用):模式选择→扫描进度→结果弹层
struct DuplicateScanView: View {
    @EnvironmentObject var app: AppModel
    @Environment(\.dismiss) private var dismiss
    @State private var mode = "similar"
    @State private var scanning = false
    @State private var progress: (done: Int, total: Int)?
    @State private var groups: [(keys: [String], scores: [Float])] = []
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
                    Text("覆盖已索引的照片(\(photoCount) 张);删除走系统相册「最近删除」,30 天内可恢复。雷同模式下分数只是相似度(92%-98%),不一定是同一张——请逐组确认。")
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
                let scored = DuplicateScanner.scan(
                    keys: data.keys, flat: data.flat, dim: data.dim, threshold: threshold) { done, total in
                    Task { @MainActor in progress = (done, total) }
                }
                let keyGroups = scored
                    .filter { $0.count > 1 }
                    .map { g in (keys: g.map { data.keys[$0.index] },
                                 scores: g.map { $0.score }) }
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

/// 查重结果:分组展示雷同照片,每张标注与组内基准照片的相似度(%)。
/// 删除分两级:每组内「删除本组所选」逐组处理(推荐,雷同≠相同);
/// 顶部「删除全部所选」跨组批量(默认每组保留第 1 张),风险自负有二次确认。
/// 删除走系统相册「最近删除」(30 天可恢复),同时清除索引。
struct DuplicateResultsView: View {
    let initialGroups: [(keys: [String], scores: [Float])]
    @EnvironmentObject var app: AppModel
    @Environment(\.dismiss) private var dismiss
    @State private var groups: [(keys: [String], scores: [Float])] = []
    @State private var assets: [String: PHAsset] = [:]
    @State private var selected: Set<String> = []
    @State private var deleting = false
    @State private var confirmDelete = false
    @State private var confirmGroup: Int?

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
                    Button(selected.isEmpty ? "勾选待删" : "删除全部所选(\(selected.count))") {
                        confirmDelete = true
                    }
                    .disabled(selected.isEmpty || deleting)
                }            }
            .confirmationDialog("删除全部 \(selected.count) 张所选照片?",
                                isPresented: $confirmDelete, titleVisibility: .visible) {
                Button("全部删除(30 天内可恢复)", role: .destructive) {
                    performDelete(ids: Array(selected))
                }
                Button("取消", role: .cancel) {}
            } message: {
                Text("跨全部组批量删除,范围大;建议逐组用「删除本组所选」更稳妥")
            }
            .confirmationDialog("删除本组所选?",
                                isPresented: Binding(get: { confirmGroup != nil },
                                                     set: { if !$0 { confirmGroup = nil } }),
                                titleVisibility: .visible) {
                Button("删除(30 天内可恢复)", role: .destructive) {
                    if let gi = confirmGroup { deleteInGroup(gi) }
                    confirmGroup = nil
                }
                Button("取消", role: .cancel) {}
            } message: {
                if let gi = confirmGroup, groups.indices.contains(gi) {
                    let picked = groups[gi].keys.filter { selected.contains($0) }.count
                    Text("只删第 \(gi + 1) 组勾选的 \(picked) 张")
                }
            }
            .onAppear {
                guard groups.isEmpty else { return }
                groups = initialGroups
                for g in groups { selected.formUnion(g.keys.dropFirst()) }   // 默认每组保留第 1 张
                let ids = Array(Set(groups.flatMap { $0.keys }))
                let fetch = PHAsset.fetchAssets(withLocalIdentifiers: ids, options: nil)
                fetch.enumerateObjects { a, _, _ in assets[a.localIdentifier] = a }
                // 已不在相册里的(被手动删过)从分组剔除
                groups = groups.map { g in
                    let keep = g.keys.enumerated().filter { assets[$0.element] != nil }
                    return (keys: keep.map(\.element),
                            scores: keep.compactMap { g.scores.indices.contains($0.offset) ? g.scores[$0.offset] : 0 })
                }.filter { $0.keys.count > 1 }
                selected = selected.filter { assets[$0] != nil }
            }
        }
    }

    private func groupSection(_ gi: Int) -> some View {
        let g = groups[gi]
        let picked = g.keys.filter { selected.contains($0) }.count
        return Section("第 \(gi + 1) 组 · \(g.keys.count) 张 · 已选删 \(picked) 张") {
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(Array(g.keys.enumerated()), id: \.element) { i, key in
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
                        .overlay(alignment: .bottomLeading) {
                            // 相似度角标:基准=组锚,其余=与锚的余弦(雷同≠相同,看分数再删)
                            Text(i == 0 ? "基准" : String(format: "%.0f%%", g.scores.indices.contains(i) ? g.scores[i] * 100 : 0))
                                .font(.caption2).monospacedDigit()
                                .padding(.horizontal, 4).padding(.vertical, 2)
                                .background(.black.opacity(0.6), in: RoundedRectangle(cornerRadius: 4))
                                .foregroundStyle(.white)
                        }
                        .onTapGesture { toggle(key) }
                    }
                }
                .padding(.vertical, 4)
            }
            Button("保留第 1 张,选中本组其余") {
                for key in g.keys.dropFirst() { selected.insert(key) }
                for key in g.keys.prefix(1) { selected.remove(key) }
            }
            .font(.footnote)
            if picked > 0 {
                Button("删除本组所选(\(picked))", role: .destructive) {
                    confirmGroup = gi
                }
                .font(.footnote)
                .disabled(deleting)   // 删除进行中禁一切删除入口,防连删
            }
        }
    }

    private func toggle(_ key: String) {
        if selected.contains(key) { selected.remove(key) } else { selected.insert(key) }
    }

    /// 组级删除:只删该组勾选的,组内剩 <2 张时整组收摊
    private func deleteInGroup(_ gi: Int) {
        let ids = groups[gi].keys.filter { selected.contains($0) }
        performDelete(ids: ids)
    }

    private func performDelete(ids: [String]) {
        deleting = true
        let ids = Array(Set(ids))
        let fetch = PHAsset.fetchAssets(withLocalIdentifiers: ids, options: nil)
        Task {
            do {
                try await PHPhotoLibrary.shared().performChanges {
                    PHAssetChangeRequest.deleteAssets(fetch as NSFastEnumeration)
                }
                for id in ids { try? app.store.removeEverythingForRef(id) }
                await MainActor.run {
                    groups = groups.map { g in
                        (keys: g.keys.filter { !ids.contains($0) },
                         scores: g.scores.enumerated().filter { !ids.contains(g.keys[$0.offset]) }.map(\.element))
                    }.filter { $0.keys.count > 1 }
                    selected.subtract(ids)
                    deleting = false
                    app.notify("已删除 \(ids.count) 张,最近删除保留 30 天")
                    // 删除后留在结果页:可继续处理/删除另外一组,退出始终由用户点「关闭」;
                    // 全部组处理完显示空态页(也不再自动退出,用户 2026-10-13 二次澄清)
                }
            } catch {
                await MainActor.run { deleting = false; app.fail(error) }
            }
        }
    }
}
