import SwiftUI
import Photos
import UniformTypeIdentifiers

struct LibraryView: View {
    @EnvironmentObject var app: AppModel

    @State private var photoCount = 0
    @State private var videoCount = 0
    @State private var importedFiles: [ImportedFile] = []
    @State private var showFilePicker = false
    @State private var showFolderPicker = false
    @State private var confirmRebuild = false
    @State private var refreshPending = false
    @State private var filesExpanded = false
    @State private var previewFile: ImportedFile?
    @State private var fileSelectionMode = false
    @State private var selectedFileIDs: Set<String> = []
    /// 删除确认弹窗的目标快照:按下「删除」那一刻锁定。弹窗存续期间选择集
    /// 可能被清空(索引进度触发的刷新/取消选择/上一次删除的收尾),标题与
    /// 动作都以快照为准——用户实测弹窗开着却显示「删除 0 个导入文件」、
    /// 按钮按了没反应(guard targets.isEmpty 静默返回)
    @State private var pendingDeleteIDs: Set<String> = []
    @State private var showFileDeleteDialog = false
    @State private var showDupResults = false
    @State private var showDupScan = false
    @State private var showPeople = false
    @State private var faceCount = 0

    private let columns = [GridItem(.flexible()), GridItem(.flexible())]

    var body: some View {
        NavigationStack {
            List {
                permissionSection
                statsSection
                indexingSection
                dedupSection
                peopleSection
                importSection
                if !importedFiles.isEmpty {
                    importedSection
                }
            }
            .navigationTitle("资料库")
            .onAppear(perform: refresh)
            .onChange(of: app.indexing.phase) { _ in refresh() }
            .fileImporter(isPresented: $showFilePicker,
                          allowedContentTypes: [.item],
                          allowsMultipleSelection: true) { result in
                handleImport(result)
            }
            .fileImporter(isPresented: $showFolderPicker,
                          allowedContentTypes: [.folder],
                          allowsMultipleSelection: false) { result in
                handleImport(result)
            }
            .confirmationDialog("全量重建会删除现有索引并重新处理所有内容,耗时较长。继续?",
                                isPresented: $confirmRebuild, titleVisibility: .visible) {
                Button("全量重建", role: .destructive) { app.indexing.rebuildAll() }
                Button("取消", role: .cancel) {}
            }
            .sheet(item: $previewFile) { f in
                NavigationStack {
                    FilePreviewScreen(record: f)
                        .toolbar {
                            ToolbarItem(placement: .topBarTrailing) {
                                Button("完成") { previewFile = nil }
                            }
                        }
                }
            }
            .sheet(isPresented: $showDupScan) {
                DuplicateScanView()
            }
            .sheet(isPresented: $showPeople) {
                PeopleView()
            }
            .confirmationDialog("删除 \(pendingDeleteIDs.count) 个导入文件", isPresented: $showFileDeleteDialog,
                                titleVisibility: .visible) {
                Button("删除记录(保留原文件)", role: .destructive) { deleteSelectedFiles(deleteOriginal: false) }
                Button("删除记录+原文件(原文件将被移除)", role: .destructive) { deleteSelectedFiles(deleteOriginal: true) }
                Button("取消", role: .cancel) {}
            }
        }
    }

    // MARK: - 权限

    @ViewBuilder
    private var permissionSection: some View {
        Section("系统相册") {
            switch app.photoLib.authStatus {
            case .notDetermined:
                Button {
                    Task { await app.photoLib.requestAccess() }
                } label: {
                    Label("授权访问全部照片与视频", systemImage: "lock.open")
                        .bold()
                }
            case .limited:
                Label("仅授权了部分照片,建议在系统设置中改为「所有照片」", systemImage: "exclamationmark.triangle")
                    .font(.footnote)
            case .authorized:
                Label("已授权,新照片会自动进入索引", systemImage: "checkmark.seal")
                    .font(.footnote).foregroundStyle(.secondary)
            default:
                Label("相册访问受限,请到系统设置中开启", systemImage: "lock")
                    .font(.footnote)
                if let url = URL(string: UIApplication.openSettingsURLString) {
                    Link("打开系统设置", destination: url)
                }
            }
        }
    }

    // MARK: - 统计

    private func statCard(_ title: String, _ value: String, _ icon: String) -> some View {
        VStack(spacing: 6) {
            Image(systemName: icon).font(.title3).foregroundStyle(.tint)
            Text(value).font(.headline)
            Text(title).font(.caption2).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 12)
        .background(Color(.systemGray6), in: RoundedRectangle(cornerRadius: 12))
    }

    private var statsSection: some View {
        Section {
            LazyVGrid(columns: columns, spacing: 10) {
                statCard("照片", "\(photoCount)", "photo")
                statCard("视频", "\(videoCount)", "video")
                statCard("导入文件", "\(importedFiles.count)", "doc")
                statCard("索引总数", "\(photoCount + videoCount + importedFiles.count)",
                         "point.3.filled.connected.trianglepath.dotted")
            }
            .listRowInsets(EdgeInsets(top: 8, leading: 12, bottom: 8, trailing: 12))
        }
    }

    // MARK: - 索引

    private var indexingSection: some View {
        Section("索引") {
            if app.indexing.isRunning {
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text(app.indexing.phaseText).font(.footnote)
                        Spacer()
                        Text("\(app.indexing.processed) / \(app.indexing.total)")
                            .font(.footnote.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }
                    ProgressView(value: app.indexing.total > 0
                                 ? Double(app.indexing.processed) / Double(app.indexing.total) : 0)
                    if app.indexing.errorCount > 0 {
                        Text("\(app.indexing.errorCount) 项处理失败(已跳过)\(app.indexing.firstErrorMessage.map { ":\($0)" } ?? "")")
                            .font(.caption2).foregroundStyle(.orange)
                    }
                    Button("停止", role: .destructive) { app.indexing.cancel() }
                        .font(.footnote)
                }
            } else {
                Button {
                    app.indexing.runIncremental()
                } label: {
                    Label("同步新增内容", systemImage: "arrow.triangle.2.circlepath")
                }
                Button {
                    confirmRebuild = true
                } label: {
                    Label("全量重建索引", systemImage: "arrow.counterclockwise")
                }
                if let last = app.indexing.lastSyncAt {
                    Text("上次同步:\(last.formatted(date: .abbreviated, time: .shortened))")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
        }
    }

    /// 批量删除选中的导入文件:记录+全类型向量+OCR 原文;
    /// deleteOriginal=true 时连书签指向的实体文件一并移除
    private func deleteSelectedFiles(deleteOriginal: Bool) {
        let targets = importedFiles.filter { pendingDeleteIDs.contains($0.id) }
        guard !targets.isEmpty else { return }
        var ok = 0
        var originals = 0
        for f in targets {
            if deleteOriginal, let url = try? app.imports.resolveURL(f) {
                do {
                    try FileManager.default.removeItem(at: url)
                    originals += 1
                } catch { /* 原文件删不掉(只读/iCloud)则仅删记录 */ }
                url.stopAccessingSecurityScopedResource()
            }
            do { try app.imports.remove(f); ok += 1 } catch { app.fail(error) }
        }
        selectedFileIDs.removeAll()
        pendingDeleteIDs.removeAll()
        fileSelectionMode = false
        refresh()
        app.notify(deleteOriginal
                   ? "已删除 \(ok) 个文件记录(原文件移除 \(originals) 个)"
                   : "已删除 \(ok) 个文件记录(原文件保留)")
    }

    // MARK: - 查重

    private var dedupSection: some View {
        Section("查重") {
            Button {
                showDupScan = true
            } label: {
                Label(photoCount < 2 ? "查重(需先索引照片)" : "查找重复/雷同照片",
                      systemImage: "square.stack.3d.down.right")
            }
            .disabled(photoCount < 2)
            Text("基于语义向量找出画面雷同的照片;删除走系统相册「最近删除」,30 天内可恢复。")
                .font(.caption2).foregroundStyle(.secondary)
        }
    }

    // MARK: - 人物归组

    private var peopleSection: some View {
        Section("人物") {
            Button {
                showPeople = true
            } label: {
                Label(faceCount > 0 ? "按人物浏览/归组(\(faceCount) 张照片)" : "按人物浏览/归组",
                      systemImage: "person.2")
            }
            .disabled(faceCount < 2)
            Text(faceCount > 0
                 ? "同一人的照片自动归为一组;给组命名后,搜索名字即可找到全部照片。"
                 : "需先在「设置」导入含人脸模型的新模型包,并点「同步新增内容」补全人脸数据(约 30-60 分钟,可中断可续)。")
                .font(.caption2).foregroundStyle(.secondary)
        }
    }

    // MARK: - 导入

    private var importSection: some View {
        Section("从「文件」导入") {
            Button {
                showFilePicker = true
            } label: {
                Label("选择文件(可多选)", systemImage: "plus.square.on.square")
            }
            Button {
                showFolderPicker = true
            } label: {
                Label("导入整个文件夹", systemImage: "folder.badge.plus")
            }
            Text("支持图片、视频、PDF、文本、代码等;文件保留在原位置,通过书签访问,不复制不占用额外空间。")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    @ViewBuilder
    private var importedSection: some View {
        Section {
            if importedFiles.count > 8 || fileSelectionMode {
                HStack {
                    Button {
                        withAnimation { filesExpanded.toggle() }
                    } label: {
                        HStack(spacing: 4) {
                            Text(filesExpanded ? "收起列表" : "展开全部 \(importedFiles.count) 个文件")
                                .font(.footnote)
                            Image(systemName: filesExpanded ? "chevron.up" : "chevron.down")
                                .font(.caption2)
                        }
                    }
                    Spacer()
                    if !fileSelectionMode {
                        Button("选择") { fileSelectionMode = true }
                            .font(.footnote)
                    }
                }
                .buttonStyle(.plain)
            }
            if filesExpanded || importedFiles.count <= 8 {
                if fileSelectionMode {
                    HStack {
                        Button("全选") {
                            selectedFileIDs = Set(importedFiles.map(\.id))
                        }
                        Spacer()
                        Button("取消选择") {
                            fileSelectionMode = false
                            selectedFileIDs.removeAll()
                        }
                        Button("删除(\(selectedFileIDs.count))", role: .destructive) {
                            pendingDeleteIDs = selectedFileIDs   // 弹窗存续期以快照为准
                            showFileDeleteDialog = true
                        }
                        .disabled(selectedFileIDs.isEmpty)
                    }
                    .font(.footnote)
                }
                ForEach(importedFiles) { file in
                    Button {
                        if fileSelectionMode {
                            if selectedFileIDs.contains(file.id) {
                                selectedFileIDs.remove(file.id)
                            } else {
                                selectedFileIDs.insert(file.id)
                            }
                        } else {
                            previewFile = file
                        }
                    } label: {
                        HStack(spacing: 10) {
                            if fileSelectionMode {
                                Image(systemName: selectedFileIDs.contains(file.id)
                                      ? "checkmark.circle.fill" : "circle")
                                    .foregroundStyle(selectedFileIDs.contains(file.id) ? .blue : .secondary)
                            }
                            FileThumbView(record: file)
                                .frame(width: 40, height: 40)
                                .clipShape(RoundedRectangle(cornerRadius: 6))
                            VStack(alignment: .leading, spacing: 2) {
                                Text(file.name).font(.footnote).lineLimit(1)
                                Text(ByteCountFormatter.string(fromByteCount: file.size, countStyle: .file)
                                     + (file.indexed ? " · 已索引" : " · 待索引"))
                                    .font(.caption2).foregroundStyle(.secondary)
                            }
                            Spacer()
                            if !fileSelectionMode {
                                Image(systemName: "doc.text.magnifyingglass")
                                    .font(.caption).foregroundStyle(.tertiary)
                            }
                        }
                    }
                    .buttonStyle(.plain)
                }
                .onDelete { indexSet in
                    for i in indexSet {
                        try? app.imports.remove(importedFiles[i])
                    }
                    refresh()
                }
            } else {
                Text("列表已收起,点上方展开")
                    .font(.caption).foregroundStyle(.secondary)
            }
        } header: {
            Button {
                withAnimation { filesExpanded.toggle() }
            } label: {
                HStack(spacing: 4) {
                    Text("已导入文件(\(importedFiles.count))")
                    Image(systemName: filesExpanded ? "chevron.down" : "chevron.right")
                        .font(.caption2)
                }
            }
            .buttonStyle(.plain)
        }
    }

    // MARK: - 动作

    private func refresh() {
        // SwiftUI 26 对「视图更新进行中写 @State」直接断言崩溃(EXC_BREAKPOINT):
        // 导入完成后 alert 绑定翻转 + onChange(phase) 同步刷新正撞在更新中途
        // (21:32 构建 69 崩溃栈:body witness→refresh)。一律推迟到本轮更新
        // 结束后执行,短时间连发合并成一次。
        guard !refreshPending else { return }
        refreshPending = true
        DispatchQueue.main.async { [self] in
            refreshPending = false
            let counts = (try? app.store.counts()) ?? [:]
            photoCount = counts[.photo] ?? 0
            videoCount = counts[.videoFrame] ?? 0
            faceCount = app.store.faceStats().photos
            importedFiles = app.imports.allFiles()
        }
    }

    private func handleImport(_ result: Result<[URL], Error>) {
        switch result {
        case .success(let urls):
            // 文件选择器回调上下文里同步做重活(枚举+写库)会在 iOS 26 崩溃,
            // 导入搬到后台线程,主线程只刷新界面
            Task.detached(priority: .userInitiated) {
                do {
                    let n = try app.imports.importURLs(urls)
                    await MainActor.run {
                        // 成功提示走「完成」弹窗,别占用 errorMessage(顶着「出错了」标题吓人)
                        if n > 0 { app.notify("已加入 \(n) 个文件,正在后台建立索引…") }
                        refresh()
                    }
                    // IndexingCoordinator 是 @MainActor:detached 上下文里必须显式跳回主线程,
                    // 否则 startRun 在后台线程执行(@Published 跨线程写、Task 丢失 actor 上下文)
                    if n > 0 { await MainActor.run { app.indexing.runIncremental() } }
                } catch {
                    await MainActor.run { app.fail(error) }
                }
            }
        case .failure(let error):
            app.fail(error)
        }
    }
}
