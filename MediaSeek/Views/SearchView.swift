import SwiftUI
import Photos

struct SearchView: View {
    @EnvironmentObject var app: AppModel

    @State private var query = ""
    @State private var scope: SearchScope = .all
    @State private var results: [DisplayHit] = []
    @State private var searching = false
    @State private var searchedOnce = false
    @State private var elapsedMs = 0
    @State private var selectedHit: DisplayHit?
    @State private var exporting = false
    @State private var selectionMode = false
    @State private var selected = Set<String>()
    @State private var batchTagText = ""
    @State private var showTagAlert = false
    @State private var showDeleteDialog = false
    @State private var showRenameAlert = false
    @State private var renameText = ""
    @State private var colorFilter: String?

    private let exampleQueries = ["一只猫", "海边的日落", "文字:身份证", "语义:身份证", "and:2026 屏幕", "or:猫 狗"]

    private let columns = [GridItem(.adaptive(minimum: 105), spacing: 10)]

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                searchField
                scopePicker
                if !app.models.anyReady {
                    modelBanner
                }
                statusLine
                resultGrid
            }
            .navigationTitle("智搜")
            .sheet(item: $selectedHit) { hit in
                DetailSheet(hit: hit)
            }
            .alert("批量标注", isPresented: $showTagAlert) {
                TextField("标签内容(如:程小姐)", text: $batchTagText)
                Button("添加") { batchTag() }
                Button("取消", role: .cancel) { batchTagText = "" }
            } message: {
                Text("将为选中的 \(selected.count) 个项目添加标签")
            }
            .confirmationDialog("删除 \(selected.count) 个选中项", isPresented: $showDeleteDialog, titleVisibility: .visible) {
                Button("从系统相册删除(移入最近删除)", role: .destructive) { deleteFromPhotos() }
                Button("仅从索引移除(下次同步会回来)", role: .destructive) { removeFromIndexOnly() }
                Button("取消", role: .cancel) {}
            } message: {
                Text("删除会同时移除其索引与标签")
            }
            .alert("重命名导入文件", isPresented: $showRenameAlert) {
                TextField("新名称(不含扩展名)", text: $renameText)
                Button("改名") { doRename() }
                Button("取消", role: .cancel) {}
            }
        }
    }

    private var searchField: some View {
        HStack(spacing: 8) {
            Image(systemName: "magnifyingglass").foregroundStyle(.secondary)
            TextField("用一句话找照片、视频、文件…", text: $query)
                .submitLabel(.search)
                .onSubmit(runSearch)
            if !query.isEmpty {
                Button {
                    query = ""
                    results = []
                    searchedOnce = false
                } label: {
                    Image(systemName: "xmark.circle.fill").foregroundStyle(.secondary)
                }
            }
        }
        .padding(10)
        .background(Color(.systemGray6), in: RoundedRectangle(cornerRadius: 12))
        .padding(.horizontal)
        .padding(.top, 8)
    }

    private var scopePicker: some View {
        Picker("范围", selection: $scope) {
            ForEach(SearchScope.allCases) { s in
                Text(s.rawValue).tag(s)
            }
        }
        .pickerStyle(.segmented)
        .padding(.horizontal)
        .padding(.vertical, 8)
    }

    private var modelBanner: some View {
        HStack(spacing: 8) {
            Image(systemName: "brain")
            Text("模型未就绪,请到「设置」导入模型包")
                .font(.footnote)
        }
        .frame(maxWidth: .infinity)
        .padding(10)
        .background(Color.orange.opacity(0.15), in: RoundedRectangle(cornerRadius: 10))
        .padding(.horizontal)
    }

    @ViewBuilder
    private var statusLine: some View {
        if searching {
            ProgressView("正在检索…").padding(.vertical, 6)
        } else if query.isEmpty {
            VStack(alignment: .leading, spacing: 10) {
                if !app.recentSearches.isEmpty {
                    HStack {
                        Text("最近搜索")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Spacer()
                        Button("清空") { app.clearSearchHistory() }
                            .font(.caption)
                    }
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 8) {
                            ForEach(app.recentSearches, id: \.self) { q in
                                Button {
                                    query = q
                                    runSearch()
                                } label: {
                                    Text(q)
                                        .font(.footnote)
                                        .padding(.horizontal, 12)
                                        .padding(.vertical, 6)
                                        .background(Color(.systemGray5), in: Capsule())
                                }
                                .buttonStyle(.plain)
                            }
                        }
                    }
                }
                HStack {
                    Text("示例")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .padding(.horizontal)
            .padding(.vertical, 6)
        } else if searchedOnce {
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 10) {
                    if selectionMode {
                        Text("已选 \(selected.count) 项")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    } else {
                        Text(colorFilter == nil
                             ? "找到 \(results.count) 个结果 · \(elapsedMs) ms"
                             : "\(visibleResults.count) / \(results.count) 个结果 · \(elapsedMs) ms")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }
                    if !results.isEmpty {
                        Spacer()
                        if exporting {
                            ProgressView()
                        } else {
                            actionsMenu
                        }
                    }
                }
                if !results.isEmpty && !selectionMode && colorFilter == nil {
                    Text("百分比 = 相对相关度(第一名 100%),非绝对概率")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
            }
            .padding(.horizontal)
            .padding(.vertical, 6)
            if app.indexing.isRunning {
                Text("⚠️ 索引重建中 \(app.indexing.processed)/\(app.indexing.total) · 当前结果暂不完整")
                    .font(.caption2)
                    .foregroundStyle(.orange)
                    .padding(.horizontal)
            }
            if !results.isEmpty && !selectionMode {
                colorFilterBar
            }
        } else {
            chips
        }
    }

    private var chips: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(exampleQueries, id: \.self) { q in
                    Button(q) {
                        query = q
                        runSearch()
                    }
                    .font(.footnote)
                    .padding(.horizontal, 12).padding(.vertical, 6)
                    .background(Color(.systemGray6), in: Capsule())
                }
            }
            .padding(.horizontal)
        }
        .padding(.bottom, 4)
    }

    /// 空结果提示:按查询模式分别说明
    private func emptyStateHint(_ query: String) -> String {
        let q = query.lowercased()
        if q.hasPrefix("语义:") {
            return "语义通道未命中(已索引 \(app.indexing.indexedPhotoCount) 张)\n试试 文字:身份证 用文字精确查找\n或换更具体的词;索引重建完成后召回会增加"
        }
        if q.hasPrefix("文字:") {
            return "文字通道未命中:该文字未被识别到\n或这些照片还没重建索引(已索引 \(app.indexing.indexedPhotoCount) 张)"
        }
        return "已索引 \(app.indexing.indexedPhotoCount) 张照片\n文字条件需要该文字被识别到;索引进行中时结果不完整,跑完再试\n也可以换个说法再搜,或用 文字:/语义: 前缀精确控制"
    }

    private var resultGrid: some View {
        ScrollView {
            if results.isEmpty && searchedOnce && !searching {
                ContentUnavailableView("没有找到相关内容",
                                       systemImage: "questionmark.folder",
                                       description: Text(emptyStateHint(query)))
                    .padding(.top, 60)
            } else {
                LazyVGrid(columns: columns, spacing: 12) {
                    ForEach(visibleResults) { hit in
                        Button {
                            if selectionMode {
                                if selected.contains(hit.id) { selected.remove(hit.id) } else { selected.insert(hit.id) }
                            } else {
                                selectedHit = hit
                            }
                        } label: {
                            HitCell(hit: hit, selectionMode: selectionMode, isSelected: selected.contains(hit.id))
                        }
                        .buttonStyle(.plain)
                    }
                }
                .padding(.horizontal)
            }
        }
    }

    private func exportToAlbum() {
        guard !exporting, !results.isEmpty else { return }
        exporting = true
        let q = query.trimmingCharacters(in: .whitespaces)
        Task {
            do {
                let n = try await SearchExporter.saveToAlbum(title: q, hits: visibleResults)
                await MainActor.run { app.notify("已把 \(n) 个项目加入相册「搜索·\(q)」\n在 系统相册 App → 我的相册 里查看") }
            } catch {
                await MainActor.run { app.fail(error) }
            }
            exporting = false
        }
    }

    private func exportToFiles() {
        guard !exporting, !results.isEmpty else { return }
        exporting = true
        let q = query.trimmingCharacters(in: .whitespaces)
        Task {
            do {
                let (n, url) = try await SearchExporter.exportToFiles(
                    folderName: q, hits: visibleResults,
                    photo: app.photoLib, imports: app.imports)
                await MainActor.run { app.notify("已导出 \(n) 个文件\n位置:「文件」App → 智搜 → 导出 → \(url.lastPathComponent)") }
            } catch {
                await MainActor.run { app.fail(error) }
            }
            exporting = false
        }
    }

    private var selectedHits: [DisplayHit] { results.filter { selected.contains($0.id) } }

    private var visibleResults: [DisplayHit] {
        guard let cf = colorFilter else { return results }
        return results.filter { $0.color == cf }
    }

    private func colorFor(_ name: String) -> Color {
        switch name {
        case "红": return .red
        case "橙": return .orange
        case "黄": return .yellow
        case "绿": return .green
        case "青": return .cyan
        case "蓝": return .blue
        case "紫": return .purple
        case "粉": return .pink
        case "黑": return .black
        case "灰": return .gray
        case "白": return .white
        default: return .clear
        }
    }

    private var colorFilterBar: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(PhotoColor.buckets, id: \.self) { name in
                    Button {
                        colorFilter = colorFilter == name ? nil : name
                    } label: {
                        HStack(spacing: 4) {
                            Circle()
                                .fill(colorFor(name))
                                .frame(width: 10, height: 10)
                            Text(name)
                                .font(.caption)
                            if let n = results.filter({ $0.color == name }).count as Int?, n > 0 {
                                Text("\(n)")
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                            }
                        }
                        .padding(.horizontal, 10)
                        .padding(.vertical, 5)
                        .background(
                            colorFilter == name ? Color.blue.opacity(0.18) : Color(.systemGray6),
                            in: Capsule())
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.horizontal)
        }
    }

    private var renameTarget: (hitId: String, record: ImportedFile)? {
        guard selectionMode else { return nil }
        let fileHits = selectedHits.compactMap { hit -> (String, ImportedFile)? in
            if case .file(let f) = hit.target { return (hit.id, f) }
            return nil
        }
        guard fileHits.count == 1, let only = fileHits.first else { return nil }
        return (only.0, only.1)
    }

    private var actionsMenu: some View {
        Menu {
            if selectionMode {
                Button { toggleSelectAll() } label: {
                    Label(selected.count == results.count ? "取消全选" : "全选", systemImage: "checkmarks")
                }
                Button { showTagAlert = true } label: { Label("批量标注…", systemImage: "tag") }
                Button { exportToAlbum(selectedOnly: true) } label: { Label("选中项存入相册", systemImage: "photo.stack") }
                Button { exportToFiles(selectedOnly: true) } label: { Label("选中项导出到文件", systemImage: "folder") }
                Button(role: .destructive) { showDeleteDialog = true } label: { Label("删除…", systemImage: "trash") }
                if renameTarget != nil {
                    Button { showRenameAlert = true; renameText = "" } label: { Label("改名(导入文件)", systemImage: "pencil") }
                }
                Divider()
                Button { selectionMode = false; selected.removeAll() } label: { Label("完成", systemImage: "checkmark") }
            } else {
                Button { selectionMode = true; selected.removeAll() } label: { Label("选择", systemImage: "checkmark.circle") }
                Button { exportToAlbum(selectedOnly: false) } label: { Label("结果存入相册", systemImage: "photo.stack") }
                Button { exportToFiles(selectedOnly: false) } label: { Label("结果导出到文件", systemImage: "folder") }
            }
        } label: {
            Label("操作", systemImage: "ellipsis.circle")
                .font(.footnote)
        }
    }

    private func toggleSelectAll() {
        if selected.count == results.count { selected.removeAll() } else { selected = Set(results.map(\.id)) }
    }

    private func exportToAlbum(selectedOnly: Bool) {
        let hits = selectedOnly ? selectedHits : visibleResults
        guard !exporting, !hits.isEmpty else { return }
        exporting = true
        let q = query.trimmingCharacters(in: .whitespaces)
        Task {
            do {
                let n = try await SearchExporter.saveToAlbum(title: q, hits: hits)
                await MainActor.run { app.notify("已把 \(n) 个项目加入相册「搜索·\(q)」\n在 系统相册 App → 我的相册 里查看") }
            } catch {
                await MainActor.run { app.fail(error) }
            }
            exporting = false
        }
    }

    private func exportToFiles(selectedOnly: Bool) {
        let hits = selectedOnly ? selectedHits : visibleResults
        guard !exporting, !hits.isEmpty else { return }
        exporting = true
        let q = query.trimmingCharacters(in: .whitespaces)
        Task {
            do {
                let (n, url) = try await SearchExporter.exportToFiles(
                    folderName: q, hits: hits,
                    photo: app.photoLib, imports: app.imports)
                await MainActor.run { app.notify("已导出 \(n) 个文件\n位置:「文件」App → 智搜 → 导出 → \(url.lastPathComponent)") }
            } catch {
                await MainActor.run { app.fail(error) }
            }
            exporting = false
        }
    }

    private func batchTag() {
        let tag = batchTagText.trimmingCharacters(in: .whitespacesAndNewlines)
        let refs = selectedHits.map { $0.refKey }
        guard !tag.isEmpty, !refs.isEmpty else { return }
        showTagAlert = false
        exporting = true
        Task {
            var ok = 0
            for ref in refs {
                do { try await app.addUserTag(refKey: ref, tag: tag); ok += 1 } catch { await MainActor.run { app.fail(error) } }
            }
            await MainActor.run { app.notify("已为 \(ok) 个项目添加标签「\(tag)」"); exporting = false }
        }
    }

    private func deleteFromPhotos() {
        let hits = selectedHits
        let ids = SearchExporter.assetLocalIDs(from: hits)
        guard !ids.isEmpty else { removeFromIndexOnly(); return }
        exporting = true
        Task {
            do {
                let fetch = PHAsset.fetchAssets(withLocalIdentifiers: ids, options: nil)
                try await PHPhotoLibrary.shared().performChanges {
                    PHAssetChangeRequest.deleteAssets(fetch as NSFastEnumeration)
                }
                for hit in hits { try? app.store.removeEverythingForRef(hit.refKey) }
                await MainActor.run {
                    results.removeAll { selected.contains($0.id) }
                    selected.removeAll()
                    selectionMode = false
                    app.notify("已删除 \(ids.count) 个项目,可在「最近删除」保留 30 天")
                }
            } catch {
                await MainActor.run { app.fail(error) }
            }
            exporting = false
        }
    }

    private func removeFromIndexOnly() {
        for hit in selectedHits { try? app.store.removeEverythingForRef(hit.refKey) }
        results.removeAll { selected.contains($0.id) }
        selected.removeAll()
        selectionMode = false
        app.notify("已从索引移除(下次同步会重新索引)")
    }

    private func doRename() {
        guard let target = renameTarget else { return }
        let newName = renameText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !newName.isEmpty else { return }
        let record = target.record
        let hitId = target.hitId
        exporting = true
        Task {
            do {
                try app.imports.rename(record, to: newName)
                await MainActor.run {
                    if let idx = results.firstIndex(where: { $0.id == hitId }) {
                        let old = results[idx]
                        results[idx] = DisplayHit(id: old.id, refKey: old.refKey, target: old.target,
                                                  kind: old.kind, title: newName,
                                                  score: old.score, date: old.date,
                                                  color: old.color)
                    }
                    app.notify("已重命名")
                }
            } catch {
                await MainActor.run { app.fail(error) }
            }
            exporting = false
        }
    }

    private func runSearch() {
        let q = query.trimmingCharacters(in: .whitespaces)
        guard !q.isEmpty else { return }
        app.addSearchHistory(q)
        searching = true
        searchedOnce = true
        Task.detached(priority: .userInitiated) { [scope] in
            var hits: [DisplayHit] = []
            var ms = 0
            do {
                let start = Date()
                hits = try await app.search.search(q, scope: scope)
                ms = Int(Date().timeIntervalSince(start) * 1000)
            } catch {
                await app.fail(error)
            }
            await MainActor.run {
                results = hits
                elapsedMs = ms
                searching = false
            }
        }
    }
}
