import Foundation
import Photos
import AVFoundation

/// 把搜索结果批量落地:系统相册专辑 或 「文件」App 文件夹
enum SearchExporter {
    private static func sanitize(_ name: String) -> String {
        let bad = CharacterSet(charactersIn: "/\\:*?\"<>|")
        let cleaned = name.components(separatedBy: bad).joined(separator: "-").trimmingCharacters(in: .whitespaces)
        return String((cleaned.isEmpty ? "未命名" : cleaned).prefix(60))
    }

    static func assetLocalIDs(from hits: [DisplayHit]) -> [String] {
        hits.compactMap { hit in
            if case .asset(let a) = hit.target { return a.localIdentifier }
            return nil
        }
    }

    // MARK: - 存入系统相册(自动创建/复用同名专辑)

    static func saveToAlbum(title: String, hits: [DisplayHit]) async throws -> Int {
        let ids = assetLocalIDs(from: hits)
        guard !ids.isEmpty else { throw MSError("结果中没有系统相册的照片/视频") }
        let albumTitle = "搜索·\(sanitize(title))"

        func findAlbum() -> PHAssetCollection? {
            let lists = PHAssetCollection.fetchAssetCollections(with: .album, subtype: .any, options: nil)
            var found: PHAssetCollection?
            lists.enumerateObjects { c, _, _ in
                if c.localizedTitle == albumTitle { found = c }
            }
            return found
        }

        var album = findAlbum()
        if album == nil {
            var placeholder: String?
            try await PHPhotoLibrary.shared().performChanges {
                let req = PHAssetCollectionChangeRequest.creationRequestForAssetCollection(withTitle: albumTitle)
                placeholder = req.placeholderForCreatedAssetCollection.localIdentifier
            }
            guard let pid = placeholder else { throw MSError("创建相册失败") }
            album = PHAssetCollection.fetchAssetCollections(withLocalIdentifiers: [pid], options: nil).firstObject
        }
        guard let target = album else { throw MSError("相册不可用") }

        let fetch = PHAsset.fetchAssets(withLocalIdentifiers: ids, options: nil)
        final class FailBox { var on = false }
        let fail = FailBox()
        try await PHPhotoLibrary.shared().performChanges {
            // iOS 26 上相册修改请求可能返回 nil(静默 no-op),记下失败,块外显式报错
            guard let req = PHAssetCollectionChangeRequest(for: target) else {
                fail.on = true
                return
            }
            req.addAssets(fetch as NSFastEnumeration)
        }
        if fail.on { throw MSError("相册「\(albumTitle)」无法修改,请重试") }
        return fetch.count
    }

    // MARK: - 导出到「文件」App(文件 → 智搜 → 导出/<名字>/)

    static func exportToFiles(folderName: String, hits: [DisplayHit],
                              photo: PhotoLibraryService,
                              imports: ImportLibrary) async throws -> (count: Int, url: URL) {
        let base = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("导出", isDirectory: true)
            .appendingPathComponent(sanitize(folderName), isDirectory: true)
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)

        var count = 0
        for hit in hits {
            if Task.isCancelled { break }
            switch hit.target {
            case .asset(let asset):
                try await exportAsset(asset, to: base, photo: photo)
                count += 1
            case .file(let record):
                let url = try imports.resolveURL(record)
                defer { url.stopAccessingSecurityScopedResource() }
                let dest = uniqueURL(base.appendingPathComponent(record.name))
                try FileManager.default.copyItem(at: url, to: dest)
                count += 1
            }
        }
        return (count, base)
    }

    private static func uniqueURL(_ url: URL) -> URL {
        var candidate = url
        var n = 1
        while FileManager.default.fileExists(atPath: candidate.path) {
            let name = url.deletingPathExtension().lastPathComponent
            let ext = url.pathExtension
            candidate = url.deletingLastPathComponent()
                .appendingPathComponent("\(name)-\(n)" + (ext.isEmpty ? "" : ".\(ext)"))
            n += 1
        }
        return candidate
    }

    private static func exportAsset(_ asset: PHAsset, to folder: URL, photo: PhotoLibraryService) async throws {
        if asset.mediaType == .image {
            // 优先取原片文件直拷
            let opts = PHContentEditingInputRequestOptions()
            opts.isNetworkAccessAllowed = true
            opts.canHandleAdjustmentData = { _ in false }
            let originalURL: URL? = await withCheckedContinuation { cont in
                asset.requestContentEditingInput(with: opts) { input, _ in
                    cont.resume(returning: input?.fullSizeImageURL)
                }
            }
            if let originalURL {
                let dest = uniqueURL(folder.appendingPathComponent(originalURL.lastPathComponent))
                try FileManager.default.copyItem(at: originalURL, to: dest)
                return
            }
            // 兜底:渲染导出 JPEG
            let data: Data? = await withCheckedContinuation { cont in
                let ropts = PHImageRequestOptions()
                ropts.isNetworkAccessAllowed = true
                ropts.deliveryMode = .highQualityFormat
                PHImageManager.default().requestImageDataAndOrientation(for: asset, options: ropts) { data, _, _, _ in
                    cont.resume(returning: data)
                }
            }
            guard let data else { throw MSError("照片导出失败") }
            let dest = uniqueURL(folder.appendingPathComponent("photo-\(asset.localIdentifier.suffix(6)).jpg"))
            try data.write(to: dest)
        } else if asset.mediaType == .video {
            let url: URL? = await withCheckedContinuation { cont in
                let vopts = PHVideoRequestOptions()
                vopts.isNetworkAccessAllowed = true
                vopts.version = .original
                PHImageManager.default().requestAVAsset(forVideo: asset, options: vopts) { av, _, _ in
                    cont.resume(returning: (av as? AVURLAsset)?.url)
                }
            }
            guard let url else { throw MSError("视频导出失败") }
            let dest = uniqueURL(folder.appendingPathComponent(url.lastPathComponent))
            try FileManager.default.copyItem(at: url, to: dest)
        }
    }
}
