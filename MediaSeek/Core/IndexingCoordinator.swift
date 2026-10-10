import Foundation
import Photos
import AVFoundation
import UIKit
import Vision
import PDFKit
import UniformTypeIdentifiers

/// 索引编排器:照片→SigLIP 图向量 + Vision 标签(走 EmbeddingGemma 中文语义),
/// 视频→关键帧向量,文件→文件名/文本块向量。后台顺序处理,实时进度。
@MainActor
final class IndexingCoordinator: ObservableObject {
    enum Phase: Equatable {
        case idle, waitingModel, photos, videos, files, done, failed(String)
    }

    @Published private(set) var phase: Phase = .idle
    @Published private(set) var processed = 0
    @Published private(set) var total = 0
    @Published private(set) var isRunning = false
    @Published private(set) var errorCount = 0
    @Published private(set) var firstErrorMessage: String?
    @Published private(set) var lastSyncAt: Date?

    /// 跨并发记录第一个处理错误(线程安全)
    final class FirstErrorBox: @unchecked Sendable {
        private let lock = NSLock()
        private var message: String?
        func set(_ m: String) {
            lock.lock()
            if message == nil { message = m }
            lock.unlock()
        }
        func get() -> String? {
            lock.lock(); defer { lock.unlock() }
            return message
        }
    }

    var videoFramesPerVideo: Int {
        get { UserDefaults.standard.object(forKey: "videoFrames") as? Int ?? 3 }
        set { UserDefaults.standard.set(newValue, forKey: "videoFrames") }
    }

    /// 中文语义标签(Vision+Gemma):增强中文物体查询,但每张照片多两次推理,慢 2-3 倍
    /// 已索引照片数(空态提示用)
    var indexedPhotoCount: Int { store.countPhotos() }

    var chineseLabels: Bool {
        get { UserDefaults.standard.object(forKey: "chineseLabels") as? Bool ?? true }
        set { UserDefaults.standard.set(newValue, forKey: "chineseLabels") }
    }

    private var processedLocal = 0

    private let store: VectorStore
    private let models: ModelManager
    private let photo: PhotoLibraryService
    private let imports: ImportLibrary
    private var task: Task<Void, Never>?

    init(store: VectorStore, models: ModelManager, photo: PhotoLibraryService, imports: ImportLibrary) {
        self.store = store
        self.models = models
        self.photo = photo
        self.imports = imports
    }

    var phaseText: String {
        switch phase {
        case .idle: return "未开始"
        case .waitingModel: return "等待模型加载…"
        case .photos: return "正在索引照片"
        case .videos: return "正在索引视频"
        case .files: return "正在索引文件"
        case .done: return "索引完成"
        case .failed(let m): return "失败:\(m)"
        }
    }

    // MARK: - 入口

    func startAutoSyncIfAuthorized() {
        guard photo.isAuthorized else { return }
        runIncremental()
    }

    func runIncremental() {
        startRun(full: false)
    }

    func rebuildAll() {
        startRun(full: true)
    }

    func cancel() {
        task?.cancel()
        task = nil
    }

    private func startRun(full: Bool) {
        guard !isRunning else { return }
        isRunning = true
        processed = 0
        processedLocal = 0
        total = 0
        errorCount = 0
        firstErrorMessage = nil
        phase = models.bothReady ? .photos : .waitingModel
        let errors = FirstErrorBox()
        task = Task { [full, errors] in
            await self.run(full: full, errors: errors)
            self.isRunning = false
            self.task = nil
        }
    }

    private func run(full: Bool, errors: FirstErrorBox) async {
        do {
            let ready = await models.waitReady()
            guard ready else {
                phase = .failed("模型未就绪,请先在「设置」中导入模型包")
                return
            }
            // 全量重建不再先清空旧索引:全量覆盖更新,搜索全程不断档
            // (结束时按现有相册清单自动清理已删除照片的旧行)
            try await syncPhotoLibrary(forceAll: full, errors: errors)
            try await syncImportedFiles()
            await prebuildLabelVocab()   // 重建完成后预建路由词表,首次搜索不再有一次性延迟

            lastSyncAt = Date()
            phase = .done
        } catch is CancellationError {
            phase = .idle
        } catch {
            phase = .failed(error.localizedDescription)
        }
    }

    /// 把库内去重标签全部嵌入词向量表(语义路由用),量级几百条、一次性
    private func prebuildLabelVocab() async {
        guard let gemma = models.gemma,
              let labels = try? store.allDistinctLabels(), !labels.isEmpty else { return }
        let known = Set(((try? store.labelVocab()) ?? []).map { $0.label })
        for label in labels.prefix(2000) where !known.contains(label) {
            guard let v = try? gemma.embedDocument(label) else { continue }
            try? store.saveLabelVec(label, vec: v)
        }
    }

    // MARK: - 相册

    private func syncPhotoLibrary(forceAll: Bool = false, errors: FirstErrorBox) async throws {
        let fetch = photo.fetchAllAssets()
        // 全量重建:所有照片视为新照片全量覆盖;增量同步:跳过已索引
        let known = forceAll ? Set<String>() : try store.refKeys(kinds: [.photo, .photoLabel, .userTag, .videoFrame])
        let recentlyDeleted = Self.recentlyDeletedIDs()

        var libIDs = Set<String>()
        var newPhotos: [PHAsset] = []
        var newVideos: [PHAsset] = []
        fetch.enumerateObjects { asset, _, _ in
            // 「最近删除」里的照片/视频不入库:不新增索引,旧行按 stale 一并清除
            guard recentlyDeleted.contains(asset.localIdentifier) == false else { return }
            libIDs.insert(asset.localIdentifier)
            guard !known.contains(asset.localIdentifier) else { return }
            if asset.mediaType == .image {
                newPhotos.append(asset)
            } else if asset.mediaType == .video {
                newVideos.append(asset)
            }
        }
        try store.removeRefs(kinds: [.photo, .photoLabel, .userTag, .videoFrame], notIn: libIDs)

        total += newPhotos.count + newVideos.count
        phase = .photos
        // 双路并发:解码/下载与 ANE 推理流水线重叠
        let useLabels = chineseLabels
        let clip = models.siglip
        let gemma = models.gemma
        let store = self.store
        let photo = self.photo
        await withTaskGroup(of: Int.self) { group in
            var index = 0
            let maxConcurrent = 2
            while index < min(maxConcurrent, newPhotos.count) {
                let asset = newPhotos[index]; index += 1
                group.addTask { await Self.embedPhotoWork(
                    asset: asset, photo: photo, store: store,
                    clip: clip, gemma: gemma, useLabels: useLabels, errors: errors) }
            }
            while !group.isEmpty {
                let fails = await group.next() ?? 0
                errorCount += fails
                processedLocal += 1
                if processedLocal % 5 == 0 { processed = processedLocal }
                if fails > 0, firstErrorMessage == nil {
                    firstErrorMessage = errors.get()
                }
                // 连续失败 20 张 = 系统性故障,立即停止并显示原因(不空烧全库)
                if errorCount >= 20, let msg = errors.get() {
                    phase = .failed("连续处理失败已停止:\(msg)")
                    return
                }
                if Task.isCancelled { break }
                if index < newPhotos.count {
                    let asset = newPhotos[index]; index += 1
                    group.addTask { await Self.embedPhotoWork(
                        asset: asset, photo: photo, store: store,
                        clip: clip, gemma: gemma, useLabels: useLabels, errors: errors) }
                }
            }
        }
        try Task.checkCancellation()   // 「停止」从这里立即生效
        processed = processedLocal
        phase = .videos
        for asset in newVideos {
            try Task.checkCancellation()
            await embedVideo(asset, errors: errors)
            processedLocal += 1
            processed = processedLocal
        }
    }

    /// 「最近删除」智能相册的资产 ID。系统未公开对应枚举,
    /// 通行做法:按智能相册 subtype 原始值 1000000201 识别(中英文标题兜底);
    /// 若系统未暴露该相册,返回空集,行为退化为不过滤。
    private static func recentlyDeletedIDs() -> Set<String> {
        var ids = Set<String>()
        let cols = PHAssetCollection.fetchAssetCollections(with: .smartAlbum, subtype: .any, options: nil)
        cols.enumerateObjects { col, _, _ in
            let isRD = col.assetCollectionSubtype.rawValue == 1000000201
                || col.localizedTitle == "最近删除" || col.localizedTitle == "Recently Deleted"
            guard isRD else { return }
            PHAsset.fetchAssets(in: col, options: nil).enumerateObjects { asset, _, _ in
                ids.insert(asset.localIdentifier)
            }
        }
        return ids
    }

    nonisolated private static func embedPhotoWork(
        asset: PHAsset, photo: PhotoLibraryService, store: VectorStore,
        clip: SigLIPEmbedder?, gemma: GemmaTextEmbedder?, useLabels: Bool,
        errors: FirstErrorBox) async -> Int {
        guard let clip, let gemma else { return 0 }
        do {
            // 2048px 用于 OCR 文字识别(拍屏角度/小字也尽量认出);320px 用于嵌入向量
            guard let full = try await photo.image(for: asset, maxPixel: 2048) else {
                errors.set("取图失败(可能 iCloud 未下载或存储空间不足):\(asset.localIdentifier)")
                return 1
            }
            try Task.checkCancellation()   // 取消:取图后立即中断
            let small = Self.downscaled(full, to: 320)
            let colorBucket = PhotoColor.bucket(of: small)
            let vec = try clip.embedImage(small)
            try store.upsert(kind: .photo, refKey: asset.localIdentifier, space: clip.space,
                             vector: vec, title: nil, date: asset.creationDate, color: colorBucket)

            // OCR 文字层:屏幕文字/证件名/编号可被精确检索
            if let ocr = Self.recognizeText(in: full) {
                try store.saveOCR(refKey: asset.localIdentifier, text: ocr)
            }

            guard useLabels, let labels = Self.visionLabels(cg: small), !labels.isEmpty else { return 0 }
            try Task.checkCancellation()
            let lvec = try gemma.embedDocument("photo tags: \(labels)")
            try store.upsert(kind: .photoLabel, refKey: asset.localIdentifier, space: gemma.space,
                             vector: lvec, title: labels, date: asset.creationDate)
            return 0
        } catch {
            if Task.isCancelled { return 0 }   // 用户停止不算错误
            errors.set("照片处理失败:\(error)")
            return 1
        }
    }

    private func embedVideo(_ asset: PHAsset, errors: FirstErrorBox) async {
        do {
            guard let clip = models.siglip else { return }
            let store = self.store
            let photo = self.photo
            let frameCount = videoFramesPerVideo
            try await Task.detached(priority: .utility) {
                guard let avAsset = await photo.avAsset(for: asset) else { return }
                let frames = try await FrameSampler.sample(asset: avAsset, count: frameCount)
                for (i, frame) in frames.enumerated() {
                    let vec = try clip.embedImage(frame)
                    try store.upsert(kind: .videoFrame, refKey: asset.localIdentifier, frameIndex: i,
                                     space: clip.space, vector: vec, title: nil, date: asset.creationDate,
                                     color: PhotoColor.bucket(of: frame))
                }
            }.value
        } catch is CancellationError {
            return   // 用户停止不算错误
        } catch {
            errors.set("视频处理失败:\(error)")
            errorCount += 1
        }
    }

    /// Vision 内置 1300+ 类分类器,取高置信度标签(如 cat / dog / beach)
    nonisolated private static func visionLabels(cg: CGImage) -> String? {
        let request = VNClassifyImageRequest()
        let handler = VNImageRequestHandler(cgImage: cg, options: [:])
        do {
            try handler.perform([request])
            let labels = (request.results ?? [])
                .filter { $0.confidence >= 0.3 }
                .prefix(6)
                .map { $0.identifier }
            return labels.isEmpty ? nil : labels.joined(separator: ", ")
        } catch {
            return nil
        }
    }

    /// Vision 文字识别(OCR):简繁中文+英文,精确检索层的数据来源
    nonisolated private static func recognizeText(in cg: CGImage) -> String? {
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        request.recognitionLanguages = ["zh-Hans", "zh-Hant", "en-US"]
        request.usesLanguageCorrection = false
        let handler = VNImageRequestHandler(cgImage: cg, options: [:])
        do {
            try handler.perform([request])
            let lines = (request.results as? [VNRecognizedTextObservation])?
                .compactMap { $0.topCandidates(1).first?.string } ?? []
            let joined = lines.joined(separator: "\n")
            return joined.isEmpty ? nil : String(joined.prefix(2000))
        } catch {
            return nil
        }
    }

    /// 高分辨率图降采样到嵌入模型需要的尺寸
    nonisolated private static func downscaled(_ image: CGImage, to side: Int) -> CGImage {
        let cs = CGColorSpaceCreateDeviceRGB()
        let ctx = CGContext(data: nil, width: side, height: side, bitsPerComponent: 8,
                            bytesPerRow: side * 4, space: cs,
                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        ctx.interpolationQuality = .medium
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: side, height: side))
        return ctx.makeImage()!
    }

    // MARK: - 导入文件

    private func syncImportedFiles() async throws {
        let files = imports.allFiles().filter { !$0.indexed }
        total += files.count
        phase = .files
        for file in files {
            try Task.checkCancellation()
            await embedImported(file)
            processed += 1
        }
    }

    func embedImported(_ record: ImportedFile) async {
        do {
            guard let gemma = models.gemma else { throw MSError("文本模型未就绪") }
            let clip = models.siglip
            let imports = self.imports
            let store = self.store
            let frameCount = videoFramesPerVideo
            try await Task.detached(priority: .utility) {
                let url = try imports.resolveURL(record)
                defer { url.stopAccessingSecurityScopedResource() }

                let type = UTType(filenameExtension: url.pathExtension) ?? .data
                let date = record.addedAt

                // 文件名总是嵌入(基础可检索性)
                let nameVec = try gemma.embedDocument("filename: \(record.name)")
                try store.upsert(kind: .file, refKey: record.id, space: gemma.space,
                                 vector: nameVec, title: record.name, date: date)

                if type.conforms(to: .image), let clip,
                   let cg = Self.downsampled(at: url, maxPixel: 320) {
                    let vec = try clip.embedImage(cg)
                    try store.upsert(kind: .file, refKey: record.id, space: clip.space,
                                     vector: vec, title: record.name, date: date)
                } else if type.conforms(to: .movie), let clip {
                    let frames = try await FrameSampler.sample(url: url, count: frameCount)
                    for (i, frame) in frames.enumerated() {
                        let vec = try clip.embedImage(frame)
                        try store.upsert(kind: .fileChunk, refKey: record.id, frameIndex: 1000 + i,
                                         space: clip.space, vector: vec, title: record.name, date: date)
                    }
                } else if let text = TextExtractor.extractText(at: url), !text.isEmpty {
                    for (i, chunk) in TextExtractor.chunks(of: text).enumerated() {
                        let vec = try gemma.embedDocument(chunk)
                        try store.upsert(kind: .fileChunk, refKey: record.id, frameIndex: i,
                                         space: gemma.space, vector: vec,
                                         title: String(chunk.prefix(60)), date: date)
                    }
                }
                try store.setFileIndexed(id: record.id)
            }.value
        } catch {
            errorCount += 1
        }
    }

    nonisolated private static func downsampled(at url: URL, maxPixel: CGFloat) -> CGImage? {
        let src = CGImageSourceCreateWithURL(url as CFURL, nil)
        guard let src else { return nil }
        let opts: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: Int(maxPixel)
        ]
        return CGImageSourceCreateThumbnailAtIndex(src, 0, opts as CFDictionary)
    }
}

// MARK: - 视频抽帧

enum FrameSampler {
    static func sample(asset: AVAsset, count: Int) async throws -> [CGImage] {
        let duration = try await asset.load(.duration)
        guard duration.seconds > 0, count > 0 else { return [] }
        let gen = AVAssetImageGenerator(asset: asset)
        gen.appliesPreferredTrackTransform = true
        gen.maximumSize = CGSize(width: 320, height: 320)
        gen.requestedTimeToleranceBefore = CMTime(seconds: 1, preferredTimescale: 600)
        gen.requestedTimeToleranceAfter = CMTime(seconds: 1, preferredTimescale: 600)

        let step = duration.seconds / Double(max(count, 1))
        let times: [CMTime] = (0..<count).map { i in
            let sec = min(Double(i) * step + step / 2, max(duration.seconds - 0.1, 0))
            return CMTime(seconds: sec, preferredTimescale: 600)
        }

        var frames: [CGImage] = []
        for await result in gen.images(for: times) {
            if case .success(let frame) = result {
                frames.append(frame.image)
            }
        }
        return frames
    }

    static func sample(url: URL, count: Int) async throws -> [CGImage] {
        let asset = AVURLAsset(url: url)
        return try await sample(asset: asset, count: count)
    }
}

// MARK: - 文本抽取与分块

enum TextExtractor {
    static let textExtensions: Set<String> = [
        "txt", "md", "markdown", "csv", "tsv", "json", "xml", "yaml", "yml",
        "log", "rtf", "swift", "m", "h", "c", "cpp", "py", "js", "ts", "html", "css",
        "java", "kt", "go", "rs", "sql", "ini", "conf", "plist"
    ]

    static func extractText(at url: URL) -> String? {
        let ext = url.pathExtension.lowercased()
        // 大于 2MB 的纯文本不做全文索引,避免病态文件
        if textExtensions.contains(ext) {
            guard let size = try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize, size < 2_000_000 else { return nil }
            guard let data = try? Data(contentsOf: url) else { return nil }
            if ext == "rtf" {
                guard let attr = try? NSAttributedString(
                    data: data,
                    options: [.documentType: NSAttributedString.DocumentType.rtf],
                    documentAttributes: nil) else { return nil }
                return attr.string
            }
            return String(data: data, encoding: .utf8)
                ?? String(data: data, encoding: .isoLatin1)
        }
        if ext == "pdf" {
            guard let doc = PDFDocument(url: url) else { return nil }
            var all = ""
            for i in 0..<min(doc.pageCount, 200) {
                if let page = doc.page(at: i), let s = page.string {
                    all += s + "\n"
                }
            }
            return all.isEmpty ? nil : all
        }
        return nil
    }

    /// 按段落聚合为 ~500 字块,相邻块重叠 ~80 字
    static func chunks(of text: String, maxChars: Int = 500, overlap: Int = 80) -> [String] {
        var chunks: [String] = []
        var current = ""
        for para in text.components(separatedBy: .newlines) {
            let p = para.trimmingCharacters(in: .whitespaces)
            if p.isEmpty { continue }
            if current.count + p.count + 1 > maxChars {
                if !current.isEmpty { chunks.append(current) }
                current = current.suffix(overlap) + " " + p
            } else {
                current += (current.isEmpty ? "" : " ") + p
            }
            if chunks.count >= 24 { break }   // 单文件块数上限
        }
        if !current.isEmpty { chunks.append(current) }
        return chunks
    }
}
