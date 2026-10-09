import Foundation
import Photos

enum SearchScope: String, CaseIterable, Identifiable {
    case all = "全部"
    case photo = "照片"
    case video = "视频"
    case file = "文件"
    var id: String { rawValue }
}

enum SearchTarget {
    case asset(PHAsset)        // 按 mediaType 区分照片/视频
    case file(ImportedFile)
}

struct DisplayHit: Identifiable {
    let id: String
    let refKey: String
    let target: SearchTarget
    let kind: ItemKind
    let title: String
    let score: Float
    let date: Date?
    let color: String?
}

/// 检索:查询同时嵌入 SigLIP 空间(查图/查视频)与 EmbeddingGemma 空间
/// (查文件、文本块、照片标签),合并去重后按相似度排序。
final class SearchEngine {
    private let store: VectorStore
    private let models: ModelManager
    private let photo: PhotoLibraryService
    private let imports: ImportLibrary

    init(store: VectorStore, models: ModelManager, photo: PhotoLibraryService, imports: ImportLibrary) {
        self.store = store
        self.models = models
        self.photo = photo
        self.imports = imports
    }

    func search(_ rawQuery: String, scope: SearchScope, topK: Int = 120) async throws -> [DisplayHit] {
        let query = rawQuery.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !query.isEmpty else { return [] }

        let imageKinds: [ItemKind] = scope == .all || scope == .photo
            ? [.photo] : (scope == .video ? [.videoFrame] : [])
        let gemmaKinds: [ItemKind] = {
            switch scope {
            case .all: return [.photoLabel, .userTag, .file, .fileChunk]
            case .photo: return [.photoLabel, .userTag]
            case .file: return [.file, .fileChunk]
            case .video: return [.userTag]
            }
        }()

        var hits: [SearchHit] = []
        let siglip = await MainActor.run { models.siglip }
        let gemma = await MainActor.run { models.gemma }
        if !imageKinds.isEmpty, let siglip {
            let q = try siglip.embedQuery(query)
            hits += try store.search(space: siglip.space, kinds: imageKinds, query: q,
                                     limit: topK, minScore: 0.12)
        }
        if !gemmaKinds.isEmpty, let gemma {
            let q = try gemma.embedQuery(query)
            hits += try store.search(space: gemma.space, kinds: gemmaKinds, query: q,
                                     limit: topK, minScore: 0.12)
        }

        // 同一 refKey 保留最高分(视频多帧、照片双空间、文件多块都会命中多次)
        var best: [String: SearchHit] = [:]
        for hit in hits {
            if let cur = best[hit.refKey] {
                if hit.score > cur.score { best[hit.refKey] = hit }
            } else {
                best[hit.refKey] = hit
            }
        }

        // OCR 文字精确层:编号/年份/证件文字的子串匹配(与语义通道互补,精确命中置顶)
        if scope != .file, query.count >= 2,
           let ocrRefs = try? store.searchOCR(query: query) {
            for (refKey, _) in ocrRefs where best[refKey] == nil {
                best[refKey] = SearchHit(kind: .photo, refKey: refKey, frameIndex: 0,
                                         space: "ocr", title: "含「\(query)」文字",
                                         date: nil, score: 1.0, color: nil)
            }
        }

        guard !best.isEmpty else { return [] }
        let merged = best.values.sorted { $0.score > $1.score }

        return resolve(merged)
    }

    private func resolve(_ hits: [SearchHit]) -> [DisplayHit] {
        let assetIDs = Set(hits.filter {
            $0.kind == .photo || $0.kind == .videoFrame || $0.kind == .photoLabel
        }.map { $0.refKey })

        var assets: [String: PHAsset] = [:]
        if !assetIDs.isEmpty {
            let fetch = PHAsset.fetchAssets(withLocalIdentifiers: Array(assetIDs), options: nil)
            fetch.enumerateObjects { asset, _, _ in assets[asset.localIdentifier] = asset }
        }

        var out: [DisplayHit] = []
        for hit in hits {
            switch hit.kind {
            case .photo, .photoLabel, .userTag, .videoFrame:
                guard let asset = assets[hit.refKey] else { continue }
                let name = asset.mediaType == .video ? "视频" : "照片"
                out.append(DisplayHit(id: "\(hit.kind.rawValue)-\(hit.refKey)",
                                      refKey: hit.refKey,
                                      target: .asset(asset),
                                      kind: hit.kind,
                                      title: hit.title ?? name,
                                      score: hit.score,
                                      date: asset.creationDate,
                                      color: hit.color))
            case .file, .fileChunk:
                guard let file = try? store.file(id: hit.refKey) else { continue }
                out.append(DisplayHit(id: "\(hit.kind.rawValue)-\(hit.refKey)",
                                      refKey: hit.refKey,
                                      target: .file(file),
                                      kind: hit.kind,
                                      title: hit.title ?? file.name,
                                      score: hit.score,
                                      date: hit.date,
                                      color: hit.color))
            }
        }
        return out
    }
}
