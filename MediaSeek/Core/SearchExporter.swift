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

    // MARK: - 单项物化(详情页「共享 / 存储到文件」,第五十七批)

    /// 详情页里被打开的任意内容:相册照片/视频 或 导入文件
    enum ShareItem {
        case asset(PHAsset)
        case file(ImportedFile)

        init(target: SearchTarget) {
            switch target {
            case .asset(let a): self = .asset(a)
            case .file(let f): self = .file(f)
            }
        }
    }

    /// 临时共享目录:单项共享/另存的物化落点。副本本就是一次性的
    /// (共享完成/被「文件」导出移走后即无用),放 tmp 让系统也可回收
    private static var shareDir: URL {
        let d = FileManager.default.temporaryDirectory
            .appendingPathComponent("share", isDirectory: true)
        try? FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        return d
    }

    /// 清掉旧的物化副本(>24h)。共享面板(AirDrop 等)在面板关闭后
    /// 系统可能仍在读文件,只按年龄清,绝不碰刚生成的
    static func purgeStaleShareCopies(olderThan hours: Double = 24) {
        let dir = shareDir
        guard let children = try? FileManager.default.contentsOfDirectory(
            at: dir, includingPropertiesForKeys: [.contentModificationDateKey]) else { return }
        let cutoff = Date().addingTimeInterval(-hours * 3600)
        for child in children {
            if let m = (try? child.resourceValues(forKeys: [.contentModificationDateKey]))?
                .contentModificationDate, m < cutoff {
                try? FileManager.default.removeItem(at: child)
            }
        }
    }

    /// 把任意目标物化成临时目录里的独立文件,返回其 URL。
    /// 共享面板与「存储到文件」都要文件 URL:相册资产的原件在系统照片
    /// 库内部;导入文件带安全域书签,共享/导出过程中作用域不保证仍在,
    /// 先落一个自有副本最稳。iCloud 未下载的资产会在此触发下载,耗时
    /// 由调用方以 busy 态展示
    static func materialize(_ item: ShareItem,
                            photo: PhotoLibraryService,
                            imports: ImportLibrary) async throws -> URL {
        switch item {
        case .asset(let asset):
            return try await exportAsset(asset, to: shareDir, photo: photo)
        case .file(let record):
            let url = try imports.resolveURL(record)
            defer { url.stopAccessingSecurityScopedResource() }
            let dest = uniqueURL(shareDir.appendingPathComponent(record.name))
            try FileManager.default.copyItem(at: url, to: dest)
            return dest
        }
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

    /// 导出单个相册资产到目标文件夹,返回落盘文件 URL(批量导出忽略,
    /// 单项物化「共享/存储到文件」要用)
    @discardableResult
    private static func exportAsset(_ asset: PHAsset, to folder: URL, photo: PhotoLibraryService) async throws -> URL {
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
                return dest
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
            return dest
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
            return dest
        } else if asset.mediaType == .audio {
            // 语音备忘录等音频资产:PHAssetResource 写出原始文件
            // (PHContentEditingInput 没有 audioURL——CI 编译实锤;
            // 验收发现:时间/地点浏览下音频资产会以照片姿态进结果,
            // 此前直抛「不支持的资产类型」挡住共享/另存)
            let resources = PHAssetResource.assetResources(for: asset)
            guard let res = resources.first(where: { $0.type == .audio }) else {
                throw MSError("音频导出失败(未找到原始资源)")
            }
            let name = res.originalFilename.isEmpty
                ? "audio-\(asset.localIdentifier.suffix(6)).m4a"
                : res.originalFilename
            let dest = uniqueURL(folder.appendingPathComponent(name))
            let opts = PHAssetResourceRequestOptions()
            opts.isNetworkAccessAllowed = true
            try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
                PHAssetResourceManager.default().writeData(for: res, toFile: dest, options: opts) { err in
                    if let err { cont.resume(throwing: err) } else { cont.resume() }
                }
            }
            return dest
        }
        throw MSError("不支持的资产类型(仅照片/视频/音频)")
    }
}
