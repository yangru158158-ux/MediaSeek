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
        // Gemma 语义只负责文件名/文件内容——照片标签的余弦虚高且区分度低
        // (「猫」vs「people, adult」和 vs「cat」差不了几分),会顶掉真命中
        let fileKinds: [ItemKind] = scope == .all || scope == .file ? [.file, .fileChunk] : []

        var channels: [(weight: Double, hits: [SearchHit])] = []
        let siglip = await MainActor.run { models.siglip }
        let gemma = await MainActor.run { models.gemma }

        // 视觉通道:SigLIP2 以英文图文对训练,中文查询先过内置词典
        if !imageKinds.isEmpty, let siglip {
            let q = try siglip.embedQuery(QueryUnderstanding.english(for: query) ?? query)
            channels.append((1.0, Self.dedupByRef(try store.search(
                space: siglip.space, kinds: imageKinds, query: q, limit: topK, minScore: 0.20))))
        }

        // 标签精确通道:查询词(含词典英译)与 Vision 英文标签做 token 级比对,零幻觉
        if !imageKinds.isEmpty {
            let tokens = Self.queryTokens(query: query)
            if !tokens.isEmpty, let rows = try? store.allPhotoLabels() {
                var scored: [(refKey: String, matches: Int)] = []
                for row in rows {
                    let labelTokens = Set(row.title.lowercased()
                        .components(separatedBy: CharacterSet(charactersIn: ", ")))
                    let m = tokens.intersection(labelTokens).count
                    if m > 0 { scored.append((row.refKey, m)) }
                }
                let hits = scored.sorted { $0.matches > $1.matches }.prefix(topK).map {
                    SearchHit(kind: .photo, refKey: $0.refKey, frameIndex: 0,
                              space: "label", title: nil, date: nil,
                              score: Float($0.matches), color: nil)
                }
                if !hits.isEmpty { channels.append((1.6, Array(hits))) }
            }
        }

        // 用户标签通道:中文子串精确匹配
        if scope != .file, let tagRefs = try? store.searchUserTags(query: query), !tagRefs.isEmpty {
            channels.append((1.6, tagRefs.map {
                SearchHit(kind: .photo, refKey: $0, frameIndex: 0,
                          space: "userTag", title: nil, date: nil, score: 1.0, color: nil)
            }))
        }

        // 文件语义通道:Gemma 文本嵌入查文件名/内容
        if !fileKinds.isEmpty, let gemma {
            let q = try gemma.embedQuery(query)
            channels.append((1.0, Self.dedupByRef(try store.search(
                space: gemma.space, kinds: fileKinds, query: q, limit: topK, minScore: 0.12))))
        }

        // OCR 文字精确通道:编号/年份/证件文字的子串匹配,权重最高、精确命中置顶
        var ocrRefs: [(refKey: String, text: String)] = []
        if scope == .all || scope == .photo, query.count >= 2 {
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
            for (rank, hit) in ch.hits.enumerated() { add(hit, weight: ch.weight, rank: rank) }
        }
        for (i, ref) in ocrRefs.enumerated() {
            add(SearchHit(kind: .photo, refKey: ref.refKey, frameIndex: 0,
                          space: "ocr", title: "含「\(query)」文字",
                          date: nil, score: 1.0, color: nil),
                weight: 8.0, rank: i)
        }
        guard !fused.isEmpty else { return [] }

        // 显示分 = 相对融合分(第一名 100%)
        let maxScore = fused.values.map { $0.score }.max() ?? 1.0
        let merged: [SearchHit] = fused.values.sorted { $0.score > $1.score }.map {
            SearchHit(kind: $0.hit.kind, refKey: $0.hit.refKey, frameIndex: $0.hit.frameIndex,
                      space: $0.hit.space, title: $0.hit.title, date: $0.hit.date,
                      score: Float($0.score / maxScore), color: $0.hit.color)
        }
        return resolve(merged)
    }

    /// 查询词集合:清洗后的原文分词 + 词典英译分词(供标签 token 比对)
    private static func queryTokens(query: String) -> Set<String> {
        var tokens = Set(query.lowercased()
            .components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty })
        if let en = QueryUnderstanding.english(for: query) {
            for t in en.lowercased().split(separator: " ") { tokens.insert(String(t)) }
        }
        return tokens
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
                                      color: hit.color ?? store.colorForRef(hit.refKey)))
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
