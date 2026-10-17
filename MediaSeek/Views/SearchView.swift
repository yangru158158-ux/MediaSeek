import SwiftUI
import Photos

/// 结果维度:一次搜索同时给出 全部/文字吻合/语义吻合 三套,可切换查看
enum ResultTab: Hashable {
    case all, text, semantic
}

struct SearchView: View {
    @EnvironmentObject var app: AppModel

    @State private var query = ""
    @State private var scope: SearchScope = .all
    @State private var results: [DisplayHit] = []
    @State private var searching = false
    @State private var searchedOnce = false
    @State private var elapsedMs = 0
    @State private var selectedHit: DisplayHit?
    // 「不相关」黑名单:当前查询词下被标记的结果(隐藏;「显示」可揭示,详情页可恢复)
    @State private var blockedRefs: Set<String> = []
    @State private var revealBlocked = false
    @State private var exporting = false
    @State private var selectionMode = false
    @State private var selected = Set<String>()
    // 划选(第二十批):横划过一排结果卡即批量勾选;竖向移动让给滚动
    @State private var swipe = SwipeSelectState()
    @State private var swipeFrames: [String: CGRect] = [:]
    @State private var refineMode = false
    @State private var refineBaseIDs = Set<String>()  // 拨动「结果内」瞬间锁定的搜索范围(清空输入框/改词不丢)
    @State private var lastRoundCount = 0   // 结果内搜索的上一轮结果数(计数行展示/refine 校验)
    // 结果内收窄历史:每次收窄前压栈,可逐步退回上一轮
    @State private var refineHistory: [(query: String, results: [DisplayHit], baseIDs: Set<String>, lastCount: Int, textIDs: Set<String>)] = []
    @State private var lastExecutedQuery = ""
    // 三维度结果切换:全部 / 文字吻合 / 语义吻合(结果集按命中通道归类)
    @State private var resultTab: ResultTab = .all
    @State private var textHitIDs: Set<String> = []
    @State private var timeChip: String?
    @State private var placeChip: String?
    @State private var batchTagText = ""
    @State private var showTagAlert = false
    @State private var showDeleteDialog = false
    @State private var showDupScan = false
    @State private var showPeople = false
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
                if hiddenBlockedCount > 0 {
                    blockedBanner
                }
                resultGrid
            }
            // 收键盘(第三十二批修正,用户实测点搜索框上方空白无反应):
            // tap 挂页面根 VStack——覆盖搜索框上下一切空白(按钮自吃点按
            // 不受影响);原先只挂结果列表,头部空白不在覆盖范围
            .contentShape(Rectangle())
            .onTapGesture { hideKeyboard() }
            .navigationTitle("智搜")
            // 键盘工具条(第三十三批,用户实测点导航栏空白不收——导航栏是
            // 系统地盘 App 拦不到点按):键盘弹起时键盘顶上常驻收起按钮,
            // 苹果备忘录/邮件同款;不依赖任何空白区域归属,一键必收
            .toolbar {
                ToolbarItemGroup(placement: .keyboard) {
                    Spacer()
                    Button {
                        hideKeyboard()
                    } label: {
                        Image(systemName: "keyboard.chevron.compact.down")
                    }
                }
            }
            .sheet(item: $selectedHit) { hit in
                DetailSheet(hit: hit,
                            searchTerm: lastExecutedQuery,
                            isBlocked: blockedRefs.contains(hit.refKey),
                            onBlockToggle: { blocked in
                                blocked ? markIrrelevant(hit) : unmarkIrrelevant(hit)
                            })
            }
            .sheet(isPresented: $showDupScan) {
                DuplicateScanView()
            }
            .sheet(isPresented: $showPeople) {
                PeopleView()
            }
            .alert("批量标注", isPresented: $showTagAlert) {
                TextField("标签内容(如:程小姐)", text: $batchTagText)
                Button("添加") { batchTag() }
                Button("取消", role: .cancel) { batchTagText = "" }
            } message: {
                Text("将为选中的 \(selected.count) 个项目添加标签")
            }
            .confirmationDialog("删除 \(selected.count) 个选中项", isPresented: $showDeleteDialog, titleVisibility: .visible) {
                if hasFileSelection {
                    Button("删除文件记录(保留原文件)", role: .destructive) { deleteImportedFiles(deleteOriginal: false) }
                    Button("删除文件记录+原文件(原文件将被移除)", role: .destructive) { deleteImportedFiles(deleteOriginal: true) }
                }
                if hasAssetSelection {
                    Button("删除照片(移入最近删除,30 天可恢复)", role: .destructive) { deleteFromPhotos() }
                }
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
                    textHitIDs = []
                    timeChip = nil
                    placeChip = nil
                    resultTab = .all
                    refineHistory.removeAll()
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

    private var hiddenBlockedCount: Int {
        results.filter { blockedRefs.contains($0.refKey) }.count
    }

    private var blockedBanner: some View {
        HStack {
            Text("已隐藏 \(hiddenBlockedCount) 个标记为不相关的结果")
                .font(.footnote)
                .foregroundColor(.secondary)
            Spacer()
            Button(revealBlocked ? "收起" : "显示") { revealBlocked.toggle() }
                .font(.footnote.weight(.medium))
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 4)
    }

    @ViewBuilder
    private var statusLine: some View {
        if searching {
            ProgressView("正在检索…").padding(.vertical, 6)
        } else if query.isEmpty && !searchedOnce {
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
                if !results.isEmpty {
                    Picker("", selection: $resultTab) {
                        let textN = results.filter { textHitIDs.contains($0.id) }.count
                        Text("全部 \(results.count)").tag(ResultTab.all)
                        Text("文字 \(textN)").tag(ResultTab.text)
                        Text("语义 \(results.count - textN)").tag(ResultTab.semantic)
                    }
                    .pickerStyle(.segmented)
                    .padding(.horizontal)
                    .padding(.bottom, 2)
                }
                HStack(spacing: 10) {
                    if selectionMode {
                        Text("已选 \(selected.count) 项")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    } else {
                        Text(colorFilter == nil
                             ? (refineMode
                                ? "结果内命中 \(results.count) / 上一轮 \(lastRoundCount) 个 · \(elapsedMs) ms"
                                : "找到 \(results.count) 个结果 · \(elapsedMs) ms")
                             : "\(visibleResults.count) / \(results.count) 个结果 · \(elapsedMs) ms")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    if !selectionMode {
                        // 结果内开关常驻,纯用户控制:开=只在当前结果集内搜(集合空了就诚实出 0),
                        // 关=全局搜。恢复路径=「返回」逐级退回保存的历史轮次
                        Button {
                            refineMode.toggle()
                            // 基数在拨动开关瞬间锁定:之后清空输入框、改词都不丢
                            // 锁定所见集(含颜色/维度镜头):所见即所搜
                            refineBaseIDs = refineMode ? Set(visibleResults.map(\.refKey)) : []
                            lastRoundCount = refineMode ? visibleResults.count : 0
                            if !refineMode { refineHistory.removeAll() }
                        } label: {
                            Text(refineMode ? "结果内:开" : "结果内")
                                .font(.caption)
                                .padding(.horizontal, 8).padding(.vertical, 4)
                                .background(refineMode ? Color.blue.opacity(0.18) : Color(.systemGray6),
                                            in: Capsule())
                        }
                        .buttonStyle(.plain)
                        // 收窄历史返回:逐步退回上一轮结果(可连按)
                        if refineMode && !refineHistory.isEmpty {
                            Button {
                                refineBack()
                            } label: {
                                Label("返回(\(refineHistory.count))", systemImage: "arrow.uturn.backward")
                                    .font(.caption)
                                    .padding(.horizontal, 8).padding(.vertical, 4)
                                    .background(Color(.systemGray6), in: Capsule())
                            }
                            .buttonStyle(.plain)
                        }
                    }
                    if !results.isEmpty {
                        if exporting {
                            ProgressView()
                        } else {
                            actionsMenu
                        }
                    }
                }
                if timeChip != nil || placeChip != nil {
                    HStack(spacing: 8) {
                        if let t = timeChip {
                            Label(t, systemImage: "calendar").font(.caption)
                                .padding(.horizontal, 8).padding(.vertical, 3)
                                .background(Color.blue.opacity(0.10), in: Capsule())
                        }
                        if let p = placeChip {
                            Label(p, systemImage: "mappin").font(.caption)
                                .padding(.horizontal, 8).padding(.vertical, 3)
                                .background(Color.blue.opacity(0.10), in: Capsule())
                        }
                    }
                    .padding(.horizontal)
                }
                if !results.isEmpty && !selectionMode && colorFilter == nil {
                    Text("百分比 = 相对相关度(第一名 100%),非绝对概率")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
                if refineMode && !results.isEmpty && !selectionMode {
                    Text("结果内搜索已开启:下一次搜索只在这些结果里进行")
                        .font(.caption2)
                        .foregroundStyle(.blue)
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
            // 文件卡片没有颜色数据,显示颜色条纯属陷阱(点了必 0)
            if !results.isEmpty && !selectionMode && scope != .file {
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
    private func emptyStateHint(_ query: String, refining: Bool) -> String {
        let q = query.lowercased()
        if refining {
            return "结果内搜索无命中:上一轮结果与新条件没有交集\n可点「↩返回」退回上一轮结果换词再搜\n或关闭「结果内」改回全局搜索"
        }
        if q.hasPrefix("语义:") {
            if let top = app.search.lastSemanticTopScore {
                return "语义通道最高相关度仅 \(Int(top * 100))%,未达 14% 门槛(已索引 \(app.indexing.indexedPhotoCount) 张)\n品牌词/抽象词的视觉相似度天然偏低,建议改用 文字:微信 直接搜截图上的字"
            }
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
                VStack(spacing: 14) {
                    ContentUnavailableView("没有找到相关内容",
                                           systemImage: "questionmark.folder",
                                           description: Text(emptyStateHint(query, refining: refineMode)))
                    if scope == .video, let cov = app.indexing.videoCoverageSummary {
                        Text(cov)
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                            .multilineTextAlignment(.center)
                    }
                    if refineMode {
                        Button {
                            refineMode = false
                            refineBaseIDs = []
                            lastRoundCount = 0
                            runSearch()
                        } label: {
                            Text("关闭结果内,全局重搜")
                                .font(.footnote.weight(.medium))
                        }
                        .buttonStyle(.bordered)
                    }
                }
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
                        .swipeCellFrame(id: hit.id, space: "resultGrid", enabled: selectionMode)
                        .opacity(blockedRefs.contains(hit.refKey) ? 0.35 : 1)
                        .overlay(alignment: .topLeading) {
                            if blockedRefs.contains(hit.refKey) {
                                Text("不相关")
                                    .font(.caption2.weight(.bold))
                                    .padding(.horizontal, 6).padding(.vertical, 2)
                                    .background(.red.opacity(0.85), in: Capsule())
                                    .foregroundStyle(.white)
                                    .padding(4)
                            }
                        }
                        .contextMenu {
                            // 选择模式让位(第三十一批):长按是划选的武装动作,
                            // 不弹「不相关」菜单
                            if !selectionMode, !lastExecutedQuery.isEmpty {
                                Button(role: .destructive) {
                                    markIrrelevant(hit)
                                } label: {
                                    Label("不相关:不再出现在「\(lastExecutedQuery)」", systemImage: "hand.thumbsdown")
                                }
                                if blockedRefs.contains(hit.refKey) {
                                    Button { unmarkIrrelevant(hit) } label: {
                                        Label("撤销不相关标记", systemImage: "eye")
                                    }
                                }
                            }
                        }
                    }
                }
                .coordinateSpace(name: "resultGrid")
                .swipeSelectContainer(space: "resultGrid", enabled: selectionMode,
                                      onSweep: { p in
                                          swipe.handle(p, frames: swipeFrames,
                                                       isSelected: { selected.contains($0) },
                                                       apply: { id, on in
                                                           if on { selected.insert(id) } else { selected.remove(id) }
                                                       })
                                      }, onEnd: { swipe.reset() })
                .onPreferenceChange(SwipeCellFramesKey.self) { swipeFrames = $0 }
                .padding(.horizontal)
            }
        }
        // 拖动结果列表立即收键盘(scrollDismissesKeyboard);
        // 点空白收起已挂页面根 VStack(见 body),此处不再重复挂
        .scrollDismissesKeyboard(.immediately)
    }

    /// 全局 resign first responder:收起搜索框输入法
    private func hideKeyboard() {
        UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder),
                                         to: nil, from: nil, for: nil)
    }

    /// 「不相关」标记:term=当前执行的查询词,库内成对记录,该查询下不再出现
    private func markIrrelevant(_ hit: DisplayHit) {
        let term = lastExecutedQuery.trimmingCharacters(in: .whitespaces)
        guard !term.isEmpty else { return }
        try? app.search.blockRef(term: term, refKey: hit.refKey)
        blockedRefs.insert(hit.refKey)
        app.notify("已标记不相关\n该结果不再出现在「\(lastExecutedQuery)」的搜索中\n(长按结果卡可撤销)")
    }

    private func unmarkIrrelevant(_ hit: DisplayHit) {
        let term = lastExecutedQuery.trimmingCharacters(in: .whitespaces)
        guard !term.isEmpty else { return }
        try? app.search.unblockRef(term: term, refKey: hit.refKey)
        blockedRefs.remove(hit.refKey)
        app.notify("已撤销不相关标记\n该结果重新出现在「\(lastExecutedQuery)」的搜索中")
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
        var out = results
        switch resultTab {
        case .all: break
        case .text: out = out.filter { textHitIDs.contains($0.id) }
        case .semantic: out = out.filter { !textHitIDs.contains($0.id) }
        }
        out = out.filter { !blockedRefs.contains($0.refKey) || revealBlocked }
        guard let cf = colorFilter else { return out }
        return out.filter { $0.color == cf }
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
                Button { showDupScan = true } label: { Label("查重(重复/雷同照片)…", systemImage: "square.stack.3d.down.right") }
                Button { showPeople = true } label: { Label("人物归组(按人浏览/命名)…", systemImage: "person.2") }
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
            do {
                // 批量通道:向量只嵌一次(此前逐张推理,几百张要几分钟)
                try await app.addUserTag(tag, toRefs: refs)
                await MainActor.run { app.notify("已为 \(refs.count) 个项目添加标签「\(tag)」") }
            } catch {
                await MainActor.run { app.fail(error) }
            }
            exporting = false
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

    /// 结果内搜索退回:弹出一轮历史,恢复上一次的结果/查询词/基数(可连按逐步退回)
    private func refineBack() {
        guard let prev = refineHistory.popLast() else { return }
        searching = false
        query = prev.query
        lastExecutedQuery = prev.query
        results = prev.results
        refineBaseIDs = prev.baseIDs
        lastRoundCount = prev.lastCount
        textHitIDs = prev.textIDs   // 文字命中集合同步恢复,否则返回后文字维度计数错乱
        colorFilter = nil           // 返回=完整恢复上一轮画面,不清镜头会继续滤成 0(2026-10-11 用户实测"无法返回")
        resultTab = .all
    }

    // MARK: - 导入文件删除(搜索结果里的文件卡片)

    private var selectedFileRecords: [ImportedFile] {
        selectedHits.compactMap { hit in
            if case .file(let f) = hit.target { return f }
            return nil
        }
    }
    private var hasFileSelection: Bool { !selectedFileRecords.isEmpty }
    private var hasAssetSelection: Bool { selectedHits.contains { hit in
        if case .asset = hit.target { return true }
        return false
    } }

    /// 删除导入文件记录:files 行、全类型向量、OCR 原文一次清净;
    /// deleteOriginal=true 时连原文件(书签指向的实体)一并移除
    private func deleteImportedFiles(deleteOriginal: Bool) {
        let files = selectedFileRecords
        guard !files.isEmpty else { return }
        exporting = true
        Task {
            var ok = 0
            var originals = 0
            for f in files {
                do {
                    if deleteOriginal, let url = try? app.imports.resolveURL(f) {
                        do {
                            try FileManager.default.removeItem(at: url)
                            originals += 1
                        } catch { /* 原文件删不掉(只读/iCloud 未同步)则仅删记录 */ }
                        url.stopAccessingSecurityScopedResource()
                    }
                    try app.imports.remove(f)
                    ok += 1
                } catch { await MainActor.run { app.fail(error) } }
            }
            let ids = Set(files.map(\.id))
            await MainActor.run {
                results.removeAll { selected.contains($0.id) || ids.contains($0.refKey) }
                selected.removeAll()
                selectionMode = false
                app.notify(deleteOriginal
                           ? "已删除 \(ok) 个文件记录(原文件移除 \(originals) 个)"
                           : "已删除 \(ok) 个文件记录(原文件保留)")
            }
            exporting = false
        }
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
        // 结果内 = 纯用户语义:开着就只在当前基数集合里搜,所见即所搜。
        // 基数空(上一轮 0)时继续搜就是 0,绝不静默退化成全局;恢复=「返回」或关开关
        let refining = refineMode
        searching = true
        searchedOnce = true
        // 本轮开始前的可见集(含颜色/维度镜头)——「所见即所搜」的收窄来源
        let lensHits = visibleResults
        // 新一轮从干净镜头开始:旧颜色筛选不清除会把无颜色结果(文件卡片)
        // 静默清成 0,计数有内容无,用户只能杀 App(2026-10-11 hsbc 实测)
        colorFilter = nil
        if refining {
            // 压栈保存未筛选的当前轮(返回时完整恢复)
            refineHistory.append((lastExecutedQuery, results, refineBaseIDs, lastRoundCount, textHitIDs))
            if refineHistory.count > 20 { refineHistory.removeFirst() }
            // 颜色/维度筛过的可见集就是下一轮的搜索范围
            refineBaseIDs = Set(lensHits.map(\.refKey))
            if refineBaseIDs.isEmpty {
                results = []
                elapsedMs = 0
                lastExecutedQuery = q
                textHitIDs = []
                timeChip = nil
                placeChip = nil
                resultTab = .all
                lastRoundCount = 0
                searching = false
                return
            }
        }
        Task.detached(priority: .userInitiated) { [scope, refining] in
            var hits: [DisplayHit] = []
            var outcome = SearchOutcome.empty
            var ms = 0
            do {
                let start = Date()
                outcome = try await app.search.search(q, scope: scope,
                                                      within: refining ? refineBaseIDs : nil)
                hits = outcome.hits
                ms = Int(Date().timeIntervalSince(start) * 1000)
                if refining {
                    // 硬保证:结果内搜索的输出绝不超过基数集合(数学不变量)
                    hits = hits.filter { refineBaseIDs.contains($0.refKey) }
                }
            } catch {
                await app.fail(error)
            }
            await MainActor.run {
                if refining, hits.count > refineBaseIDs.count {
                    hits = Array(hits.prefix(refineBaseIDs.count))
                }
                results = hits
                elapsedMs = ms
                lastExecutedQuery = q
                blockedRefs = app.search.blockedRefs(term: q)
                revealBlocked = false
                textHitIDs = outcome.textIDs
                timeChip = outcome.timeLabel
                placeChip = outcome.placeLabel
                resultTab = q.hasPrefix("文字:") ? .text : (q.hasPrefix("语义:") ? .semantic : .all)
                searching = false
                // 级联收窄:本轮结果(含 0)自动成为下一轮「结果内」的范围,所见即所搜;
                // 空了之后的恢复=「返回」退回保存的历史轮次,或关「结果内」改全局
                if refining {
                    refineBaseIDs = Set(hits.map(\.refKey))
                    lastRoundCount = hits.count
                }
            }
        }
    }
}
