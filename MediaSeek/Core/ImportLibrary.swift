import Foundation
import UniformTypeIdentifiers
import UIKit
import QuickLookThumbnailing

/// 从"文件"App 导入任意文件 / 整个文件夹。
/// 用安全域书签(bookmark)持久化访问权,不复制文件,不占双倍存储。
final class ImportLibrary: ObservableObject {
    private let store: VectorStore
    private let cache = NSCache<NSString, UIImage>()

    init(store: VectorStore) {
        self.store = store
    }

    // MARK: - 导入

    /// 返回新导入的文件数(文件夹会递归展开为多个文件记录)
    func importURLs(_ urls: [URL]) throws -> Int {
        var count = 0
        for url in urls {
            let scoped = url.startAccessingSecurityScopedResource()
            defer { if scoped { url.stopAccessingSecurityScopedResource() } }
            var isDir: ObjCBool = false
            guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir) else { continue }

            if isDir.boolValue {
                let folderID = try store.addFolder(bookmark: try url.bookmarkData())
                let enumerator = FileManager.default.enumerator(
                    at: url,
                    includingPropertiesForKeys: [.fileSizeKey, .isRegularFileKey],
                    options: [.skipsHiddenFiles, .skipsPackageDescendants])
                while let child = enumerator?.nextObject() as? URL {
                    let values = try? child.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
                    guard values?.isRegularFile == true else { continue }
                    let record = ImportedFile(
                        id: UUID().uuidString,
                        folderID: folderID,
                        relPath: child.path.replacingOccurrences(of: url.path + "/", with: ""),
                        bookmark: nil,
                        name: child.lastPathComponent,
                        size: Int64(values?.fileSize ?? 0),
                        addedAt: Date(),
                        indexed: false)
                    try store.addFile(record)
                    count += 1
                }
            } else {
                let size = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
                let record = ImportedFile(
                    id: UUID().uuidString,
                    folderID: nil,
                    relPath: nil,
                    bookmark: try url.bookmarkData(),
                    name: url.lastPathComponent,
                    size: Int64(size),
                    addedAt: Date(),
                    indexed: false)
                try store.addFile(record)
                count += 1
            }
        }
        return count
    }

    func allFiles() -> [ImportedFile] {
        (try? store.allFiles()) ?? []
    }

    func remove(_ record: ImportedFile) throws {
        try store.deleteFile(id: record.id)
    }

    /// 重命名导入文件(保留原扩展名),并同步索引记录
    func rename(_ record: ImportedFile, to newNameRaw: String) throws {
        let newName = newNameRaw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !newName.isEmpty else { throw MSError("名称不能为空") }
        guard !newName.contains("/") && !newName.contains("\\") else { throw MSError("名称不能包含 / 或 \\") }

        let url = try resolveURL(record)
        defer { url.stopAccessingSecurityScopedResource() }
        let ext = url.pathExtension
        let fileName = ext.isEmpty ? newName : "\(newName).\(ext)"
        let dest = url.deletingLastPathComponent().appendingPathComponent(fileName)

        if dest.standardizedFileURL.path != url.standardizedFileURL.path {
            if FileManager.default.fileExists(atPath: dest.path) {
                throw MSError("同名文件已存在:\(fileName)")
            }
            try FileManager.default.moveItem(at: url, to: dest)
        }
        try store.renameFile(id: record.id, newName: fileName)
    }

    // MARK: - 访问已导入文件

    /// 解析出可读 URL(内部处理安全域访问;调用方用完应 stopAccessing)
    func resolveURL(_ record: ImportedFile) throws -> URL {
        if let folderID = record.folderID, let relPath = record.relPath {
            let data = try store.folderBookmark(id: folderID)
            var stale = false
            let folderURL = try URL(resolvingBookmarkData: data, options: [], relativeTo: nil, bookmarkDataIsStale: &stale)
            _ = folderURL.startAccessingSecurityScopedResource()   // 文件夹书签授权整个目录树
            return folderURL.appendingPathComponent(relPath)
        }
        guard let data = record.bookmark else { throw MSError("文件记录缺少书签") }
        var stale = false
        let url = try URL(resolvingBookmarkData: data, options: [], relativeTo: nil, bookmarkDataIsStale: &stale)
        _ = url.startAccessingSecurityScopedResource()
        return url
    }

    // MARK: - 缩略图

    func thumbnail(for record: ImportedFile, pixel: CGFloat = 220) async -> UIImage? {
        let key = "\(record.id)-\(Int(pixel))" as NSString
        if let hit = cache.object(forKey: key) { return hit }
        guard let url = try? resolveURL(record) else { return nil }
        defer { url.stopAccessingSecurityScopedResource() }

        var image: UIImage?
        if let type = UTType(filenameExtension: url.pathExtension) {
            if type.conforms(to: .image) {
                image = downsampledImage(at: url, pixel: pixel).map(UIImage.init)
            }
        }
        if image == nil {
            image = await quickLookThumbnail(at: url, pixel: pixel)
        }
        if let image { cache.setObject(image, forKey: key) }
        return image
    }

    private func downsampledImage(at url: URL, pixel: CGFloat) -> CGImage? {
        let src = CGImageSourceCreateWithURL(url as CFURL, nil)
        guard let src else { return nil }
        let opts: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceShouldCacheImmediately: true,
            kCGImageSourceThumbnailMaxPixelSize: Int(pixel * 2)
        ]
        return CGImageSourceCreateThumbnailAtIndex(src, 0, opts as CFDictionary)
    }

    private func quickLookThumbnail(at url: URL, pixel: CGFloat) async -> UIImage? {
        await withCheckedContinuation { cont in
            let request = QLThumbnailGenerator.Request(
                fileAt: url,
                size: CGSize(width: pixel, height: pixel),
                scale: UIScreen.main.scale,
                representationTypes: .thumbnail)
            QLThumbnailGenerator.shared.generateBestRepresentation(for: request) { rep, _ in
                cont.resume(returning: rep?.uiImage)
            }
        }
    }
}
