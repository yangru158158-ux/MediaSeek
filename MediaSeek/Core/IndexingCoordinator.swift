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
    @Published private(set) var lastSyncAt: Date?

    var videoFramesPerVideo: Int {
        get { UserDefaults.standard.object(forKey: "videoFrames") as? Int ?? 3 }
        set { UserDefaults.standard.set(newValue, forKey: "videoFrames") }
    }

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
        total = 0
        errorCount = 0
        phase = models.bothReady ? .photos : .waitingModel
        task = Task { [full] in
            await self.run(full: full)
            self.isRunning = false
            self.task = nil
        }
    }

    private func run(full: Bool) async {
        do {
            let ready = await models.waitReady()
            guard ready else {
                phase = .failed("模型未就绪,请先在「设置」中导入模型包")
                return
            }
            if full { try store.deleteAll() }

            try await syncPhotoLibrary()
            try await syncImportedFiles()

            lastSyncAt = Date()
            phase = .done
        } catch is CancellationError {
            phase = .idle
        } catch {
            phase = .failed(error.localizedDescription)
        }
    }

    // MARK: - 相册

    private func syncPhotoLibrary() async throws {
        let fetch = photo.fetchAllAssets()
        let known = try store.refKeys(kinds: [.photo, .photoLabel, .videoFrame])

        var libIDs = Set<String>()
        var newPhotos: [PHAsset] = []
        var newVideos: [PHAsset] = []
        fetch.enumerateObjects { asset, _, _ in
            libIDs.insert(asset.localIdentifier)
            guard !known.contains(asset.localIdentifier) else { return }
            if asset.mediaType == .image {
                newPhotos.append(asset)
            } else if asset.mediaType == .video {
                newVideos.append(asset)
            }
        }
        try store.removeRefs(kinds: [.photo, .photoLabel, .videoFrame], notIn: libIDs)

        total += newPhotos.count + newVideos.count
        phase = .photos
        for asset in newPhotos {
            try Task.checkCancellation()
            await embedPhoto(asset)
            processed += 1
        }
        phase = .videos
        for asset in newVideos {
            try Task.checkCancellation()
            await embedVideo(asset)
            processed += 1
        }
    }

    private func embedPhoto(_ asset: PHAsset) async {
        do {
            guard let clip = models.siglip, let gemma = models.gemma else { return }
            let store = self.store
            let photo = self.photo
            try await Task.detached(priority: .utility) {
                guard let cg = try await photo.image(for: asset, maxPixel: 320) else { return }
                let vec = try clip.embedImage(cg)
                try store.upsert(kind: .photo, refKey: asset.localIdentifier, space: clip.space,
                                 vector: vec, title: nil, date: asset.creationDate)

                // Vision 图像分类 → 标签文本 → EmbeddingGemma 向量(支持中文语义查询)
                if let labels = Self.visionLabels(cg: cg), !labels.isEmpty {
                    let lvec = try gemma.embedDocument("photo tags: \(labels)")
                    try store.upsert(kind: .photoLabel, refKey: asset.localIdentifier, space: gemma.space,
                                     vector: lvec, title: labels, date: asset.creationDate)
                }
            }.value
        } catch {
            errorCount += 1
        }
    }

    private func embedVideo(_ asset: PHAsset) async {
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
                                     space: clip.space, vector: vec, title: nil, date: asset.creationDate)
                }
            }.value
        } catch {
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
