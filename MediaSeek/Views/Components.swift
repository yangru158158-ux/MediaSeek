import Foundation
import SwiftUI
import Photos
import AVKit
import QuickLook
import UIKit
import CoreLocation

/// 缩略图加载(相册资产):PHCachingImageManager + 内存缓存
final class PhotoThumbLoader {
    static let shared = PhotoThumbLoader()
    private let manager = PHCachingImageManager()
    private let cache = NSCache<NSString, UIImage>()

    func thumb(for asset: PHAsset, pixel: CGFloat) async -> UIImage? {
        let key = "\(asset.localIdentifier)-\(Int(pixel))" as NSString
        if let hit = cache.object(forKey: key) { return hit }
        let options = PHImageRequestOptions()
        options.deliveryMode = .highQualityFormat
        options.isNetworkAccessAllowed = true
        let image: UIImage? = await withCheckedContinuation { cont in
            var finished = false
            manager.requestImage(for: asset,
                                 targetSize: CGSize(width: pixel, height: pixel),
                                 contentMode: .aspectFill,
                                 options: options) { img, info in
                guard !finished else { return }
                // 降质帧等最终回调;iCloud 照片会先回调「图在云端」(img=nil 且非降质)
                // 再回调最终帧——无一次性守卫时两次都 resume = EXC_BREAKPOINT
                if (info?[PHImageResultIsDegradedKey] as? Bool) == true { return }
                finished = true
                cont.resume(returning: img)
            }
        }
        if let image { cache.setObject(image, forKey: key) }
        return image
    }
}

/// 相册缩略图
struct AssetThumbView: View {
    let asset: PHAsset
    var pixel: CGFloat = 220

    @State private var image: UIImage?

    var body: some View {
        ZStack {
            Rectangle().fill(Color(.systemGray5))
            if let image {
                Image(uiImage: image).resizable().scaledToFill()
            }
        }
        .clipped()
        .task(id: asset.localIdentifier) {
            image = await PhotoThumbLoader.shared.thumb(for: asset, pixel: pixel)
        }
    }
}

/// 导入文件缩略图
struct FileThumbView: View {
    let record: ImportedFile
    @EnvironmentObject var app: AppModel
    @State private var image: UIImage?

    var body: some View {
        ZStack {
            Rectangle().fill(Color(.systemGray5))
            if let image {
                Image(uiImage: image).resizable().scaledToFill()
            } else {
                Image(systemName: "doc").font(.largeTitle).foregroundStyle(.secondary)
            }
        }
        .clipped()
        .task(id: record.id) {
            image = await app.imports.thumbnail(for: record)
        }
    }
}

/// 搜索结果格
struct HitCell: View {
    let hit: DisplayHit
    var selectionMode: Bool = false
    var isSelected: Bool = false
    @EnvironmentObject var app: AppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            thumbnail
                .frame(height: 110)
                .frame(maxWidth: .infinity)
                .clipShape(RoundedRectangle(cornerRadius: 10))
                .overlay(alignment: .topLeading) { badge }
                .overlay(alignment: .topTrailing) {
                    if selectionMode {
                        Image(systemName: isSelected ? "checkmark.circle.fill" : "circle")
                            .font(.title3)
                            .foregroundStyle(isSelected ? Color.blue : .white)
                            .shadow(radius: 2)
                            .padding(4)
                    }
                }
                .overlay(
                    RoundedRectangle(cornerRadius: 10)
                        .stroke(isSelected ? Color.blue : .clear, lineWidth: 2)
                )
            Text(hit.title)
                .font(.caption2)
                .lineLimit(1)
            Text("\(Int(hit.score * 100))%")
                .font(.caption2)
                .foregroundStyle(.tertiary)
        }
    }

    @ViewBuilder
    private var thumbnail: some View {
        switch hit.target {
        case .asset(let asset):
            AssetThumbView(asset: asset)
                .overlay { hit.kind == .videoFrame || hit.kind == .videoFrameLabel ? playIcon : nil }
        case .file(let record):
            FileThumbView(record: record)
        }
    }

    private var playIcon: some View {
        ZStack {
            Circle().fill(.black.opacity(0.45)).frame(width: 30, height: 30)
            Image(systemName: "play.fill").foregroundStyle(.white).font(.caption)
        }
    }

    private var badge: some View {
        Text(badgeText)
            .font(.system(size: 9, weight: .semibold))
            .padding(.horizontal, 5).padding(.vertical, 2)
            .background(.black.opacity(0.55), in: Capsule())
            .foregroundStyle(.white)
            .padding(4)
    }

    private var badgeText: String {
        switch hit.kind {
        case .photo, .photoLabel, .face: return "照片"
        case .videoFrame, .videoFrameLabel: return "视频"
        case .file, .fileChunk: return "文件"
        case .userTag:
            if case .asset(let a) = hit.target { return a.mediaType == .video ? "视频" : "照片" }
            return "标记"
        }
    }
}

/// 详情页:照片查看 / 视频播放 / 文件 QuickLook + 元信息与跳转
struct DetailSheet: View {
    let hit: DisplayHit
    var searchTerm: String = ""
    var isBlocked: Bool = false
    var onBlockToggle: ((Bool) -> Void)? = nil
    @EnvironmentObject var app: AppModel
    @Environment(\.dismiss) private var dismiss
    @State private var confirmDelete = false
    @State private var placeText: String?
    @State private var fileInfoLines: [(String, String)] = []
    @State private var folderURL: URL?

    private var fileRecord: ImportedFile? {
        if case .file(let r) = hit.target { return r }
        return nil
    }

    private var asset: PHAsset? {
        if case .asset(let a) = hit.target { return a }
        return nil
    }

    private static let dateFmt: DateFormatter = {
        let f = DateFormatter()
        f.dateStyle = .long
        f.timeStyle = .short
        f.locale = Locale(identifier: "zh_CN")
        return f
    }()

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                content
                if let asset {
                    assetInfo(asset)
                        .padding(.horizontal, 16).padding(.vertical, 8)
                }
                if fileRecord != nil, !fileInfoLines.isEmpty {
                    fileInfo
                        .padding(.horizontal, 16).padding(.vertical, 8)
                }
                if isAsset {
                    TagEditorView(refKey: hit.refKey)
                        .padding(.vertical, 8)
                }
            }
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                if fileRecord != nil {
                    ToolbarItem(placement: .topBarLeading) {
                        Button(role: .destructive) {
                            confirmDelete = true
                        } label: {
                            Image(systemName: "trash")
                        }
                    }
                }
                if !searchTerm.isEmpty, let onBlockToggle {
                    ToolbarItem(placement: .topBarLeading) {
                        Button {
                            onBlockToggle(!isBlocked)
                        } label: {
                            Label(isBlocked ? "恢复" : "不相关",
                                  systemImage: isBlocked ? "eye" : "hand.thumbsdown")
                        }
                    }
                }
                ToolbarItem(placement: .topBarTrailing) {
                    ShareSaveButton(item: SearchExporter.ShareItem(target: hit.target))
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button("完成") { dismiss() }
                }
            }
            .confirmationDialog("删除这份导入文件?", isPresented: $confirmDelete, titleVisibility: .visible) {
                Button("删除记录(保留原文件)", role: .destructive) {
                    if let record = fileRecord { try? app.imports.remove(record) }
                    dismiss()
                }
                Button("删除记录+原文件(原文件将被移除)", role: .destructive) {
                    if let record = fileRecord {
                        if let url = try? app.imports.resolveURL(record) {
                            try? FileManager.default.removeItem(at: url)
                            url.stopAccessingSecurityScopedResource()
                        }
                        try? app.imports.remove(record)
                    }
                    dismiss()
                }
                Button("取消", role: .cancel) {}
            }
            .onAppear { loadInfo() }
        }
    }

    /// 照片/视频元信息:拍摄日期、位置、尺寸 + 在相册 App 打开
    private func assetInfo(_ a: PHAsset) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            if let d = a.creationDate {
                Text("拍摄日期:\(Self.dateFmt.string(from: d))")
            }
            Text(placeText ?? (a.location != nil ? "位置解析中…" : "无位置信息"))
            Text("尺寸:\(a.pixelWidth) × \(a.pixelHeight)")
            Button {
                UIApplication.shared.open(URL(string: "photos-redirect://") ?? URL(string: "photos://")!)
            } label: {
                Label("在「照片」App 中打开", systemImage: "photo.on.rectangle")
                    .font(.footnote.weight(.medium))
            }
        }
        .font(.footnote)
        .foregroundColor(.secondary)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    /// 导入文件元信息:大小、创建时间、所在文件夹 + 打开所在文件夹
    private var fileInfo: some View {
        VStack(alignment: .leading, spacing: 4) {
            ForEach(fileInfoLines, id: \.0) { pair in
                Text("\(pair.0):\(pair.1)")
            }
            Button {
                if let folderURL {
                    // iOS 无公开 API 定位到任意文件夹;退而求其次:拷贝完整路径 + 跳「文件」App 手动定位
                    UIPasteboard.general.string = folderURL.path
                    UIApplication.shared.open(URL(string: "shareddocuments://") ?? URL(string: "file://")!)
                }
            } label: {
                Label("拷贝路径,打开「文件」App", systemImage: "folder")
                    .font(.footnote.weight(.medium))
            }
        }
        .font(.footnote)
        .foregroundColor(.secondary)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    /// 元信息装载:资源反地理编码 / 文件的真实创建时间与所在文件夹
    private func loadInfo() {
        if let a = asset, let loc = a.location {
            CLGeocoder().reverseGeocodeLocation(loc) { marks, _ in
                let text: String
                if let m = marks?.first {
                    let parts = [m.subLocality ?? m.name, m.locality].compactMap { $0 }
                    text = parts.isEmpty
                        ? String(format: "%.4f, %.4f", loc.coordinate.latitude, loc.coordinate.longitude)
                        : parts.joined(separator: " ")
                } else {
                    text = String(format: "%.4f, %.4f", loc.coordinate.latitude, loc.coordinate.longitude)
                }
                DispatchQueue.main.async { placeText = text }
            }
        }
        guard let record = fileRecord, let url = try? app.imports.resolveURL(record) else { return }
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        folderURL = url.deletingLastPathComponent()
        var lines: [(String, String)] = [("文件名", record.name)]
        let rv = try? url.resourceValues(forKeys: [.fileSizeKey, .creationDateKey])
        if let bytes = rv?.fileSize {
            lines.append(("大小", ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .file)))
        }
        if let created = rv?.creationDate {
            lines.append(("创建时间", Self.dateFmt.string(from: created)))
        } else {
            lines.append(("导入时间", Self.dateFmt.string(from: record.addedAt)))
        }
        lines.append(("所在文件夹", url.deletingLastPathComponent().lastPathComponent))
        fileInfoLines = lines
    }

    private var isAsset: Bool {
        if case .asset = hit.target { return true }
        return false
    }

    @ViewBuilder
    private var content: some View {
        switch hit.target {
        case .asset(let asset):
            if asset.mediaType == .video {
                VideoPlayerScreen(asset: asset)
            } else {
                FullImageViewer(asset: asset)
            }
        case .file(let record):
            FilePreviewScreen(record: record)
        }
    }
}


// MARK: - 单项共享 / 存储到「文件」(第五十七批:任何入口打开的照片/视频/文件统一动作)

/// 系统共享面板。相册资产直接传 PHAsset(系统给原件、自带 iCloud 下载);
/// 导入文件传物化后的临时 URL
struct ShareSheet: UIViewControllerRepresentable {
    let items: [Any]

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }

    func updateUIViewController(_ vc: UIActivityViewController, context: Context) {}
}

/// 「存储到文件」:系统文档导出面板,用户自选目标位置。
/// 拿到的永远是物化出来的临时副本(asCopy:false=直接移走,省一次大文件
/// 拷贝);原件(相册资产/用户文件夹里的导入文件)绝不经过这条路径
struct DocumentExportPicker: UIViewControllerRepresentable {
    let url: URL

    func makeUIViewController(context: Context) -> UIDocumentPickerViewController {
        let picker = UIDocumentPickerViewController(forExporting: [url], asCopy: false)
        return picker
    }

    func updateUIViewController(_ vc: UIDocumentPickerViewController, context: Context) {}
}

/// 详情页「共享 / 存储到文件」按钮:覆盖相册照片、相册视频、导入文件。
/// 「存储到文件」需先把内容物化成文件(iCloud 大视频可能要下载几分钟),
/// 按钮转圈直到弹系统面板
struct ShareSaveButton: View {
    let item: SearchExporter.ShareItem
    @EnvironmentObject var app: AppModel
    @State private var busy = false
    @State private var shareItems: [Any]?
    @State private var showShare = false
    @State private var exportTarget: ExportTarget?

    /// 「存储到文件」目标(物化完成后才弹系统面板;URL 非 Identifiable 需包装)
    struct ExportTarget: Identifiable {
        let id = UUID()
        let url: URL
    }

    var body: some View {
        Menu {
            Button {
                share()
            } label: {
                Label("共享…", systemImage: "square.and.arrow.up")
            }
            Button {
                saveToFiles()
            } label: {
                Label("存储到「文件」…", systemImage: "folder.badge.arrow.down")
            }
        } label: {
            if busy {
                ProgressView()
            } else {
                Image(systemName: "square.and.arrow.up")
            }
        }
        .disabled(busy)
        .sheet(isPresented: $showShare) {
            if let items = shareItems {
                ShareSheet(items: items)
            }
        }
        .sheet(item: $exportTarget) { target in
            DocumentExportPicker(url: target.url)
        }
    }

    private func share() {
        guard !busy else { return }
        switch item {
        case .asset(let asset):
            // 相册资产交系统原件,无需物化
            shareItems = [asset]
            showShare = true
        case .file:
            busy = true
            Task {
                do {
                    let url = try await SearchExporter.materialize(item, photo: app.photoLib,
                                                                   imports: app.imports)
                    shareItems = [url]
                    showShare = true
                } catch {
                    app.fail(error)
                }
                busy = false
            }
        }
    }

    private func saveToFiles() {
        guard !busy else { return }
        busy = true
        Task {
            do {
                let url = try await SearchExporter.materialize(item, photo: app.photoLib,
                                                               imports: app.imports)
                exportTarget = ExportTarget(url: url)
            } catch {
                app.fail(error)
            }
            busy = false
        }
    }
}


struct FullImageViewer: View {
    let asset: PHAsset
    @State private var image: UIImage?

    var body: some View {
        GeometryReader { geo in
            ZStack {
                Color.black
                if let image {
                    Image(uiImage: image).resizable().scaledToFit()
                } else {
                    ProgressView().tint(.white)
                }
            }
            .task {
                let side = max(geo.size.width, geo.size.height) * UIScreen.main.scale
                image = await PhotoThumbLoader.shared.thumb(for: asset, pixel: side)
            }
        }
    }
}

struct VideoPlayerScreen: View {
    let asset: PHAsset
    @EnvironmentObject var app: AppModel
    @State private var player: AVPlayer?

    var body: some View {
        Group {
            if let player {
                VideoPlayer(player: player)
            } else {
                ZStack { Color.black; ProgressView().tint(.white) }
            }
        }
        .task {
            if player == nil, let item = await app.photoLib.playerItem(for: asset) {
                player = AVPlayer(playerItem: item)
                player?.play()
            }
        }
        .onDisappear { player?.pause() }
    }
}

struct FilePreviewScreen: View {
    let record: ImportedFile
    @EnvironmentObject var app: AppModel
    @State private var previewURL: URL?

    var body: some View {
        VStack(spacing: 8) {
            if let url = previewURL {
                Text(url.lastPathComponent)
                    .font(.footnote).foregroundStyle(.secondary)
                Text("点击下方「快速查看」预览文件")
                    .font(.caption).foregroundStyle(.tertiary)
            } else {
                ProgressView()
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .quickLookPreview($previewURL)
        .task {
            if previewURL == nil, let url = try? app.imports.resolveURL(record) {
                previewURL = url
            }
        }
    }
}


/// 照片/视频的人名与主题标签编辑器
struct TagEditorView: View {
    let refKey: String
    @EnvironmentObject var app: AppModel
    @State private var tags: [String] = []
    @State private var showAdd = false
    @State private var newTag = ""
    @State private var busy = false

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("我的标签(人名/主题,可用于搜索)")
                .font(.caption)
                .foregroundStyle(.secondary)
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(tags, id: \.self) { tag in
                        HStack(spacing: 4) {
                            Text(tag)
                            Button {
                                try? app.store.deleteUserTag(refKey: refKey, tag: tag)
                                load()
                            } label: {
                                Image(systemName: "xmark.circle.fill")
                                    .foregroundStyle(.secondary)
                            }
                            .buttonStyle(.plain)
                        }
                        .font(.footnote)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 5)
                        .background(Color(.systemBlue).opacity(0.12), in: Capsule())
                    }
                    Button {
                        showAdd = true
                    } label: {
                        Label("添加", systemImage: "plus.circle.fill")
                            .font(.footnote)
                    }
                    .buttonStyle(.borderless)
                    .disabled(busy)
                }
                .padding(.horizontal, 2)
            }
        }
        .padding(.horizontal)
        .task(id: refKey) { load() }
        .alert("添加标签(如:程小姐)", isPresented: $showAdd) {
            TextField("标签内容", text: $newTag)
            Button("添加") { addTag() }
            Button("取消", role: .cancel) { newTag = "" }
        } message: {
            Text("之后在搜索页输入这个标签即可找到本照片/视频")
        }
    }

    private func load() {
        tags = (try? app.store.userTags(forRefKey: refKey)) ?? []
    }

    private func addTag() {
        let t = newTag.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !t.isEmpty, !busy else { return }
        busy = true
        newTag = ""
        Task {
            do {
                try await app.addUserTag(refKey: refKey, tag: t)
                load()
            } catch {
                app.fail(error)
            }
            busy = false
        }
    }
}
