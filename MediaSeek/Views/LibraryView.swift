import SwiftUI
import Photos
import UniformTypeIdentifiers

struct LibraryView: View {
    @EnvironmentObject var app: AppModel

    @State private var photoCount = 0
    @State private var videoCount = 0
    @State private var vectorCount = 0
    @State private var importedFiles: [ImportedFile] = []
    @State private var showFilePicker = false
    @State private var showFolderPicker = false
    @State private var confirmRebuild = false

    private let columns = [GridItem(.flexible()), GridItem(.flexible())]

    var body: some View {
        NavigationStack {
            List {
                permissionSection
                statsSection
                indexingSection
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
                statCard("向量条目", "\(vectorCount)", "point.3.filled.connected.trianglepath.dotted")
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
        Section("已导入文件") {
            ForEach(importedFiles) { file in
                HStack(spacing: 10) {
                    FileThumbView(record: file)
                        .frame(width: 40, height: 40)
                        .clipShape(RoundedRectangle(cornerRadius: 6))
                    VStack(alignment: .leading, spacing: 2) {
                        Text(file.name).font(.footnote).lineLimit(1)
                        Text(ByteCountFormatter.string(fromByteCount: file.size, countStyle: .file)
                             + (file.indexed ? " · 已索引" : " · 待索引"))
                            .font(.caption2).foregroundStyle(.secondary)
                    }
                }
            }
            .onDelete { indexSet in
                for i in indexSet {
                    try? app.imports.remove(importedFiles[i])
                }
                refresh()
            }
        }
    }

    // MARK: - 动作

    private func refresh() {
        let counts = (try? app.store.counts()) ?? [:]
        photoCount = counts[.photo] ?? 0
        videoCount = counts[.videoFrame] ?? 0
        vectorCount = counts.values.reduce(0, +)
        importedFiles = app.imports.allFiles()
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
                        app.errorMessage = n > 0 ? "已加入 \(n) 个文件,正在后台建立索引…" : nil
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
