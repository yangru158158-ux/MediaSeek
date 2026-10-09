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
        let raw = rawQuery.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !raw.isEmpty else { return [] }
        let query = QueryUnderstanding.core(raw)
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

        // 各通道独立检索再按排名融合(RRF)——Gemma 余弦天然偏高(标签全 0.6+),
        // 按原始分数直接合并会压过视觉通道的真命中
        var channels: [[SearchHit]] = []
        let siglip = await MainActor.run { models.siglip }
        let gemma = await MainActor.run { models.gemma }
        if !imageKinds.isEmpty, let siglip {
            // SigLIP2 以英文图文对训练,中文查询先过内置词典
            let q = try siglip.embedQuery(QueryUnderstanding.english(for: query) ?? query)
            channels.append(Self.dedupByRef(try store.search(
                space: siglip.space, kinds: imageKinds, query: q, limit: topK, minScore: 0.10)))
        }
        if !gemmaKinds.isEmpty, let gemma {
            let q = try gemma.embedQuery(query)
            channels.append(Self.dedupByRef(try store.search(
                space: gemma.space, kinds: gemmaKinds, query: q, limit: topK, minScore: 0.12)))
        }

        // OCR 文字精确通道:编号/年份/证件文字的子串匹配,权重最高、精确命中置顶
        var ocrRefs: [(refKey: String, text: String)] = []
        if scope != .file, query.count >= 2 {
            ocrRefs = (try? store.searchOCR(query: query)) ?? []
        }

        // Reciprocal Rank Fusion:贡献 = 权重/(60+名次);OCR 通道权重 8,必压语义通道
        let K = 60.0
        var fused: [String: (hit: SearchHit, score: Double)] = [:]
        func add(_ hit: SearchHit, weight: Double, rank: Int) {
            let s = weight / (K + Double(rank + 1))
            if let cur = fused[hit.refKey] {
                // 标题优先取语义通道命中(「含…文字」标记只留给纯 OCR 命中)
                let winner: SearchHit
                if cur.hit.space == "ocr", hit.space != "ocr" { winner = hit }
                else { winner = hit.score > cur.hit.score ? hit : cur.hit }
                fused[hit.refKey] = (winner, cur.score + s)
            } else {
                fused[hit.refKey] = (hit, s)
            }
        }
        for ch in channels {
            for (rank, hit) in ch.enumerated() { add(hit, weight: 1.0, rank: rank) }
        }
        for (i, ref) in ocrRefs.enumerated() {
            add(SearchHit(kind: .photo, refKey: ref.refKey, frameIndex: 0,
                          space: "ocr", title: "含「\(query)」文字",
                          date: nil, score: 1.0, color: nil),
                weight: 8.0, rank: i)
        }
        guard !fused.isEmpty else { return [] }

        // 显示分 = 相对融合分(第一名 100%),替代原先满屏 66% 的原始量纲
        let maxScore = fused.values.map { $0.score }.max() ?? 1.0
        let merged: [SearchHit] = fused.values.sorted { $0.score > $1.score }.map {
            SearchHit(kind: $0.hit.kind, refKey: $0.hit.refKey, frameIndex: $0.hit.frameIndex,
                      space: $0.hit.space, title: $0.hit.title, date: $0.hit.date,
                      score: Float($0.score / maxScore), color: $0.hit.color)
        }
        return resolve(merged)
    }

    /// 通道内按 refKey 去重(视频多帧/文件多块),供 RRF 按名次计分
    private static func dedupByRef(_ hits: [SearchHit]) -> [SearchHit] {
        var best: [String: SearchHit] = [:]
        for hit in hits {
            if let cur = best[hit.refKey] {
                if hit.score > cur.score { best[hit.refKey] = hit }
            } else {
                best[hit.refKey] = hit
            }
        }
        return best.values.sorted { $0.score > $1.score }
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
