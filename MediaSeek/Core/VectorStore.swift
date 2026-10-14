import Foundation
import SQLite3
import Accelerate

/// 媒体条目类别。photoLabel 是照片的 Vision 标签文本向量(gemma 空间),
/// 让"一只猫"这类中文查询通过多语言文本模型命中照片。
enum ItemKind: String, CaseIterable {
    case photo, photoLabel, userTag, videoFrame, videoFrameLabel, file, fileChunk, face
}

struct SearchHit {
    let kind: ItemKind
    let refKey: String
    let frameIndex: Int
    let space: String
    let title: String?
    let date: Date?
    let score: Float
    let color: String?
}

/// 从"文件"App 导入的文件记录(书签持久化,不复制文件本身)
struct ImportedFile: Identifiable {
    let id: String
    var folderID: Int64?      // 非空表示位于某个导入的文件夹内
    var relPath: String?
    var bookmark: Data?       // 独立导入时的文件书签
    var name: String
    var size: Int64
    var addedAt: Date
    var indexed: Bool
}

/// 轻量 SQLite 向量库:每行一个归一化 float32 向量,检索用 vDSP 暴力点积
/// (768 维 × 数万条 < 10ms),足够"快速找到所有相关图片"。
final class VectorStore {
    private var db: OpaquePointer?
    private static let transient = unsafeBitCast(OpaquePointer(bitPattern: -1),
                                                 to: sqlite3_destructor_type.self)

    /// 数据库并发防护:FULLMUTEX 只保证单条 API 调用原子,跨线程的
    /// 事务/多语句序列仍会交错(构建 68 导入崩溃的归因)。所有公开方法
    /// 经串行队列执行;已在队列上的调用直接执行,事务嵌套不死锁。
    private static let dbKey = DispatchSpecificKey<UInt8>()
    private let dbQueue: DispatchQueue
    private var onDbQueue: Bool { DispatchQueue.getSpecific(key: Self.dbKey) == 1 }
    private func serialized<T>(_ work: () throws -> T) rethrows -> T {
        if onDbQueue { return try work() }
        return try dbQueue.sync(execute: work)
    }

    init() {
        dbQueue = DispatchQueue(label: "com.medseek.vectordb", qos: .userInitiated)
        dbQueue.setSpecific(key: Self.dbKey, value: 1)
    }

    // MARK: - 打开 / 建表

    func open() throws { try serialized {
        guard db == nil else { return }
        let dir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("MediaSeek", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let path = dir.appendingPathComponent("index.sqlite").path
        var handle: OpaquePointer?
        guard sqlite3_open_v2(path, &handle,
                              SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX,
                              nil) == SQLITE_OK else {
            throw MSError("无法打开数据库 \(path)")
        }
        db = handle
        try exec("PRAGMA journal_mode=WAL")
        try exec("PRAGMA synchronous=NORMAL")
        try exec("CREATE TABLE IF NOT EXISTS ocr_text(ref_key TEXT PRIMARY KEY, text TEXT)")
        // 人脸处理记账:该照片已按当前人脸模型做过检测/嵌入(含 0 脸照片)。
        // 没有它,无脸照片(截图/风景)会因「缺 face 行」被每轮同步反复重处理
        try exec("CREATE TABLE IF NOT EXISTS face_done(ref_key TEXT PRIMARY KEY)")
        // 人工排除的照片:用户在人物归组里勾选「移出本组」的错误分组照片,
        // 全部人脸不参与聚类(153 用户要求:人工排除个别错误分组)
        try exec("CREATE TABLE IF NOT EXISTS face_block(ref_key TEXT PRIMARY KEY)")
        // 一次性迁移:132 版人脸对齐几何错误(变换顺序反+y 轴未翻转),全部对齐图
        // 是黑图、人脸嵌入彼此相同(36 张脸归一组实锤)——清空脸向量与记账,
        // 下一次同步自动全量重嵌(走修复后的对齐)
        if !UserDefaults.standard.bool(forKey: "face_align_v2") {
            try? exec("DELETE FROM items WHERE kind = 'face'")
            try? exec("DELETE FROM face_done")
            UserDefaults.standard.set(true, forKey: "face_align_v2")
        }
        // 一次性迁移 v3:135 版未按 x 排序眼位,若 Vision 的 leftEye 命名与图像
        // 左右相反,对齐脸倒立、嵌入区分度崩塌(不同人+证件聚 128 张大组)——再清一次
        if !UserDefaults.standard.bool(forKey: "face_align_v3") {
            try? exec("DELETE FROM items WHERE kind = 'face'")
            try? exec("DELETE FROM face_done")
            UserDefaults.standard.set(true, forKey: "face_align_v3")
        }
        // 一次性迁移 v4:关键点坐标一直按「整图归一化」换算,而 Vision 的
        // normalizedPoints 实为「人脸框内归一化」——眼睛被算到脸框外,错法随
        // 脸在照片里的位置/大小变化,嵌入近乎随机(同人 0.24 < 跨人 0.56,
        // CI 真实照片端到端实锤)。修的是坐标换算、模型文件未变,指纹不会
        // 触发重建,必须迁移清库重嵌
        if !UserDefaults.standard.bool(forKey: "face_align_v4") {
            try? exec("DELETE FROM items WHERE kind = 'face'")
            try? exec("DELETE FROM face_done")
            UserDefaults.standard.set(true, forKey: "face_align_v4")
        }
        // 「不相关」黑名单:term=归一化查询词,ref=被标记的结果(该查询下不再出现)
        try exec("CREATE TABLE IF NOT EXISTS query_block(term TEXT NOT NULL, ref_key TEXT NOT NULL, created_at REAL NOT NULL, UNIQUE(term, ref_key))")
        try exec("CREATE TABLE IF NOT EXISTS label_vocab(label TEXT PRIMARY KEY, vec BLOB)")
        try exec("""
        CREATE TABLE IF NOT EXISTS items(
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          kind TEXT NOT NULL,
          ref_key TEXT NOT NULL,
          frame_index INTEGER NOT NULL DEFAULT 0,
          space TEXT NOT NULL,
          dim INTEGER NOT NULL,
          vec BLOB NOT NULL,
          title TEXT,
          created_at REAL,
          color TEXT,
          UNIQUE(kind, ref_key, frame_index)
        )
        """)
        // 旧库补列迁移:必须放在建表之后——放前面的话,新库上表还不存在,
        // ALTER 被 try? 静默吞掉,color 列就永远缺失(导致全部照片索引失败的事故)
        try? exec("ALTER TABLE items ADD COLUMN color TEXT")
        try exec("CREATE INDEX IF NOT EXISTS idx_items_kind_space ON items(kind, space)")
        try exec("""
        CREATE TABLE IF NOT EXISTS folders(
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          bookmark BLOB NOT NULL,
          added_at REAL NOT NULL
        )
        """)
        try exec("""
        CREATE TABLE IF NOT EXISTS files(
          id TEXT PRIMARY KEY,
          folder_id INTEGER,
          rel_path TEXT,
          bookmark BLOB,
          name TEXT NOT NULL,
          size INTEGER DEFAULT 0,
          added_at REAL NOT NULL,
          indexed INTEGER NOT NULL DEFAULT 0
        )
        """)
        // 自愈:清剿漏绑 id 时代写入的 NULL 毒行(记录已无主,重导即重建)
        try? exec("DELETE FROM files WHERE id IS NULL")
        // 自愈:已入库但从未做过 OCR 的 PDF 置回待索引,下次同步逐页 OCR
        // (扫描件月结单此前只有文件名可搜;有 ocr_text 原文的不再折腾)
        try? exec("""
        UPDATE files SET indexed = 0
        WHERE indexed = 1 AND lower(name || COALESCE(rel_path,'')) LIKE '%.pdf'
          AND id NOT IN (SELECT ref_key FROM ocr_text)
        """)
        // 自愈:文本/表格文件在「原文入 ocr_text」改造(2MB 上限放宽/XLSX 内容索引)之前
        // 已按旧逻辑索引,置回待索引补全文;新索引器对无文字文件写 "" 占位,故至多重处理一次
        try? exec("""
        UPDATE files SET indexed = 0
        WHERE indexed = 1 AND (
            lower(name || COALESCE(rel_path,'')) LIKE '%.xlsx' OR
            lower(name || COALESCE(rel_path,'')) LIKE '%.xlsm' OR
            lower(name || COALESCE(rel_path,'')) LIKE '%.csv' OR
            lower(name || COALESCE(rel_path,'')) LIKE '%.tsv' OR
            lower(name || COALESCE(rel_path,'')) LIKE '%.txt' OR
            lower(name || COALESCE(rel_path,'')) LIKE '%.md'
          ) AND id NOT IN (SELECT ref_key FROM ocr_text)
        """)
        // 一次性迁移:历史向量未归一化(点积≠余弦,门槛随查询漂移)→ 统一单位化
        if !UserDefaults.standard.bool(forKey: "vec_norm_migrated_1") {
            var updates: [(rowid: Int64, vec: Data)] = []
            let sel = try prepare("SELECT rowid, vec, dim FROM items")
            while sqlite3_step(sel) == SQLITE_ROW {
                let d = Int(sqlite3_column_int(sel, 2))
                guard d > 0, let blob = sqlite3_column_blob(sel, 1) else { continue }
                let v = Array(UnsafeBufferPointer(
                    start: blob.assumingMemoryBound(to: Float.self), count: d))
                let norm = vDSP.dot(v, v).squareRoot()
                guard norm > 0.0001, abs(norm - 1) > 0.001 else { continue }
                var unit = v.map { $0 / norm }
                updates.append((sqlite3_column_int64(sel, 0),
                                unit.withUnsafeBytes { Data($0) }))
            }
            sqlite3_finalize(sel)
            if !updates.isEmpty {
                let upd = try prepare("UPDATE items SET vec = ? WHERE rowid = ?")
                try exec("BEGIN IMMEDIATE")
                for u in updates {
                    u.vec.withUnsafeBytes { raw in
                        _ = sqlite3_bind_blob(upd, 1, raw.baseAddress, Int32(u.vec.count), Self.transient)
                    }
                    sqlite3_bind_int64(upd, 2, u.rowid)
                    sqlite3_step(upd)
                    sqlite3_reset(upd)
                    sqlite3_clear_bindings(upd)
                }
                try exec("COMMIT")
                sqlite3_finalize(upd)
            }
            UserDefaults.standard.set(true, forKey: "vec_norm_migrated_1")
        }
    } }

    private func exec(_ sql: String) throws {
        guard let db else { throw MSError("数据库未打开") }
        var err: UnsafeMutablePointer<CChar>?
        guard sqlite3_exec(db, sql, nil, nil, &err) == SQLITE_OK else {
            let msg = err.map { String(cString: $0) } ?? "SQL 错误"
            sqlite3_free(err)
            throw MSError(msg)
        }
    }

    deinit {
        if let db { sqlite3_close_v2(db) }
    }

    // MARK: - 事务

    func beginTransaction() throws { try exec("BEGIN IMMEDIATE") }
    func endTransaction() throws { try exec("COMMIT") }
    func rollback() { try? exec("ROLLBACK") }

    // MARK: - 向量条目

    func upsert(kind: ItemKind, refKey: String, frameIndex: Int = 0,
                space: String, vector: [Float], title: String?, date: Date?,
                color: String? = nil) throws { try serialized {
        guard let db else { throw MSError("数据库未打开") }
        do {
            try upsertRow(kind: kind, refKey: refKey, frameIndex: frameIndex, space: space,
                          vector: vector, title: title, date: date, color: color)
        } catch {
            // 自愈:旧库缺 color 列时补列并重试一次
            let msg = String(cString: sqlite3_errmsg(db))
            guard msg.contains("no column named color") else { throw MSError(msg) }
            try? exec("ALTER TABLE items ADD COLUMN color TEXT")
            try upsertRow(kind: kind, refKey: refKey, frameIndex: frameIndex, space: space,
                          vector: vector, title: title, date: date, color: color)
        }
    } }

    private func upsertRow(kind: ItemKind, refKey: String, frameIndex: Int,
                           space: String, vector: [Float], title: String?, date: Date?,
                           color: String?) throws {
        guard let db else { throw MSError("数据库未打开") }
        let sql = """
        INSERT INTO items(kind, ref_key, frame_index, space, dim, vec, title, created_at, color)
        VALUES(?,?,?,?,?,?,?,?,?)
        ON CONFLICT(kind, ref_key, frame_index) DO UPDATE SET
          space=excluded.space, dim=excluded.dim, vec=excluded.vec,
          title=excluded.title, created_at=excluded.created_at, color=excluded.color
        """
        let stmt = try prepare(sql)
        defer { sqlite3_finalize(stmt) }
        let v = vector
        let dim = v.count
        let blobSize = dim * MemoryLayout<Float>.size
        try v.withUnsafeBytes { raw in
            sqlite3_bind_text(stmt, 1, kind.rawValue, -1, Self.transient)
            sqlite3_bind_text(stmt, 2, refKey, -1, Self.transient)
            sqlite3_bind_int(stmt, 3, Int32(frameIndex))
            sqlite3_bind_text(stmt, 4, space, -1, Self.transient)
            sqlite3_bind_int(stmt, 5, Int32(dim))
            _ = raw.baseAddress.map { sqlite3_bind_blob(stmt, 6, $0, Int32(blobSize), Self.transient) }
            if let title { sqlite3_bind_text(stmt, 7, title, -1, Self.transient) }
            else { sqlite3_bind_null(stmt, 7) }
            sqlite3_bind_double(stmt, 8, date?.timeIntervalSince1970 ?? 0)
            if let color { sqlite3_bind_text(stmt, 9, color, -1, Self.transient) }
            else { sqlite3_bind_null(stmt, 9) }
            guard sqlite3_step(stmt) == SQLITE_DONE else {
                throw MSError(String(cString: sqlite3_errmsg(db)))
            }
        }
    }

    func refKeys(kinds: [ItemKind]) throws -> Set<String> { try serialized {
        guard let db else { return [] }
        let ph = kinds.map { _ in "?" }.joined(separator: ",")
        let stmt = try prepare("SELECT DISTINCT ref_key FROM items WHERE kind IN (\(ph))")
        defer { sqlite3_finalize(stmt) }
        for (i, k) in kinds.enumerated() {
            sqlite3_bind_text(stmt, Int32(i + 1), k.rawValue, -1, Self.transient)
        }
        var result: Set<String> = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            if let c = sqlite3_column_text(stmt, 0) {
                result.insert(String(cString: c))
            }
        }
        return result
    } }

    /// 库同步:删除已不在系统相册里的条目
    func removeRefs(kinds: [ItemKind], notIn keep: Set<String>) throws { try serialized {
        guard let db else { return }
        let existing = try refKeys(kinds: kinds)
        let stale = existing.subtracting(keep)
        guard !stale.isEmpty else { return }
        let ph = kinds.map { _ in "?" }.joined(separator: ",")
        let stmt = try prepare("DELETE FROM items WHERE kind IN (\(ph)) AND ref_key = ?")
        defer { sqlite3_finalize(stmt) }
        try exec("BEGIN")
        let ocrStmt = try prepare("DELETE FROM ocr_text WHERE ref_key = ?")
        defer { sqlite3_finalize(ocrStmt) }
        let faceDoneStmt = try prepare("DELETE FROM face_done WHERE ref_key = ?")
        defer { sqlite3_finalize(faceDoneStmt) }
        for key in stale {
            for (i, k) in kinds.enumerated() {
                sqlite3_bind_text(stmt, Int32(i + 1), k.rawValue, -1, Self.transient)
            }
            sqlite3_bind_text(stmt, Int32(kinds.count + 1), key, -1, Self.transient)
            sqlite3_step(stmt)
            sqlite3_reset(stmt)
            sqlite3_clear_bindings(stmt)
            sqlite3_bind_text(ocrStmt, 1, key, -1, Self.transient)
            sqlite3_step(ocrStmt)
            sqlite3_reset(ocrStmt)
            sqlite3_clear_bindings(ocrStmt)
            sqlite3_bind_text(faceDoneStmt, 1, key, -1, Self.transient)
            sqlite3_step(faceDoneStmt)
            sqlite3_reset(faceDoneStmt)
            sqlite3_clear_bindings(faceDoneStmt)
        }
        try exec("COMMIT")
    } }

    func removeByRef(kind: ItemKind, refKey: String) throws { try serialized {
        let stmt = try prepare("DELETE FROM items WHERE kind = ? AND ref_key = ?")
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_text(stmt, 1, kind.rawValue, -1, Self.transient)
        sqlite3_bind_text(stmt, 2, refKey, -1, Self.transient)
        sqlite3_step(stmt)
    } }

    func counts() throws -> [ItemKind: Int] { try serialized {
        guard let db else { return [:] }
        let stmt = try prepare("""
        SELECT kind, COUNT(DISTINCT ref_key) FROM items
        WHERE kind NOT IN ('photoLabel','userTag') GROUP BY kind
        """)
        defer { sqlite3_finalize(stmt) }
        var out: [ItemKind: Int] = [:]
        while sqlite3_step(stmt) == SQLITE_ROW {
            let k = String(cString: sqlite3_column_text(stmt, 0))
            if let kind = ItemKind(rawValue: k) {
                out[kind] = Int(sqlite3_column_int(stmt, 1))
            }
        }
        return out
    } }

    func deleteAll() throws { try serialized {
        try exec("DELETE FROM items")
        try exec("UPDATE files SET indexed = 0")
    } }

    /// 余弦检索(向量写入时已归一化,点积即余弦)
    func search(space: String, kinds: [ItemKind], query: [Float],
                limit: Int, minScore: Float, among: Set<String>? = nil) throws -> [SearchHit] { try serialized {
        guard let db else { return [] }
        let ph = kinds.map { _ in "?" }.joined(separator: ",")
        let stmt = try prepare("""
        SELECT kind, ref_key, frame_index, title, created_at, dim, vec, color
        FROM items WHERE space = ? AND kind IN (\(ph))
        """)
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_text(stmt, 1, space, -1, Self.transient)
        for (i, k) in kinds.enumerated() {
            sqlite3_bind_text(stmt, Int32(i + 2), k.rawValue, -1, Self.transient)
        }
        let q = query
        var hits: [SearchHit] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            let kind = ItemKind(rawValue: String(cString: sqlite3_column_text(stmt, 0))) ?? .file
            let refKey = String(cString: sqlite3_column_text(stmt, 1))
            // 结果内限定:只对基数集合内的行算分(全表扫描成本不变)
            if let among, !among.contains(refKey) { continue }
            let frameIndex = Int(sqlite3_column_int(stmt, 2))
            let title = sqlite3_column_type(stmt, 3) == SQLITE_NULL
                ? nil : String(cString: sqlite3_column_text(stmt, 3))
            let ts = sqlite3_column_double(stmt, 4)
            let date = ts > 0 ? Date(timeIntervalSince1970: ts) : nil
            let dim = Int(sqlite3_column_int(stmt, 5))
            guard dim == q.count, let blob = sqlite3_column_blob(stmt, 6) else { continue }
            let v = blob.assumingMemoryBound(to: Float.self)
            let score = vDSP.dot(Array(UnsafeBufferPointer(start: v, count: dim)), q)
            let color = sqlite3_column_type(stmt, 7) == SQLITE_NULL
                ? nil : String(cString: sqlite3_column_text(stmt, 7))
            if score >= minScore {
                hits.append(SearchHit(kind: kind, refKey: refKey, frameIndex: frameIndex,
                                      space: space, title: title, date: date, score: score,
                                      color: color))
            }
        }
        return Array(hits.sorted { $0.score > $1.score }.prefix(limit))
    } }

    // MARK: - 导入文件

    func addFolder(bookmark: Data) throws -> Int64 { try serialized {
        let stmt = try prepare("INSERT INTO folders(bookmark, added_at) VALUES(?,?)")
        defer { sqlite3_finalize(stmt) }
        _ = bookmark.withUnsafeBytes { sqlite3_bind_blob(stmt, 1, $0.baseAddress, Int32(bookmark.count), Self.transient) }
        sqlite3_bind_double(stmt, 2, Date().timeIntervalSince1970)
        guard sqlite3_step(stmt) == SQLITE_DONE else { throw MSError("写入文件夹记录失败") }
        return sqlite3_last_insert_rowid(db)
    } }

    func folderBookmark(id: Int64) throws -> Data { try serialized {
        let stmt = try prepare("SELECT bookmark FROM folders WHERE id = ?")
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_int64(stmt, 1, id)
        guard sqlite3_step(stmt) == SQLITE_ROW,
              let blob = sqlite3_column_blob(stmt, 0) else {
            throw MSError("找不到源文件夹,可能已被移动或删除")
        }
        return Data(bytes: blob, count: Int(sqlite3_column_bytes(stmt, 0)))
    } }

    // MARK: - 用户标签(人名/主题)

    func userTags(forRefKey refKey: String) throws -> [String] { try serialized {
        guard let db else { return [] }
        let stmt = try prepare("SELECT title FROM items WHERE kind = 'userTag' AND ref_key = ? ORDER BY id")
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_text(stmt, 1, refKey, -1, Self.transient)
        var out: [String] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            if let c = sqlite3_column_text(stmt, 0) {
                out.append(String(cString: c))
            }
        }
        return out
    } }

    /// 全部照片标签行(检索端做 token 级精确匹配用)
    func allPhotoLabels() throws -> [(refKey: String, title: String)] { try serialized {
        guard let db else { return [] }
        let stmt = try prepare("SELECT ref_key, title FROM items WHERE kind = 'photoLabel' AND title IS NOT NULL")
        defer { sqlite3_finalize(stmt) }
        var out: [(String, String)] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            out.append((String(cString: sqlite3_column_text(stmt, 0)),
                        String(cString: sqlite3_column_text(stmt, 1))))
        }
        return out.map { (refKey: $0.0, title: $0.1) }
    } }

    /// 全部视频帧标签行(视频侧的标签精确通道数据源)
    func allVideoFrameLabels() throws -> [(refKey: String, title: String)] { try serialized {
        guard let db else { return [] }
        let stmt = try prepare("SELECT ref_key, title FROM items WHERE kind = 'videoFrameLabel' AND title IS NOT NULL")
        defer { sqlite3_finalize(stmt) }
        var out: [(String, String)] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            out.append((String(cString: sqlite3_column_text(stmt, 0)),
                        String(cString: sqlite3_column_text(stmt, 1))))
        }
        return out.map { (refKey: $0.0, title: $0.1) }
    } }

    /// 用户标签子串精确匹配(中文)
    func searchUserTags(query: String) throws -> [String] { try serialized {
        guard let db, !query.isEmpty else { return [] }
        let escaped = query
            .replacingOccurrences(of: "%", with: "\\%")
            .replacingOccurrences(of: "_", with: "\\_")
        let stmt = try prepare("SELECT ref_key FROM items WHERE kind = 'userTag' AND title LIKE ? ESCAPE '\\' LIMIT 80")
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_text(stmt, 1, "%\(escaped)%", -1, Self.transient)
        var out: [String] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            out.append(String(cString: sqlite3_column_text(stmt, 0)))
        }
        return out
    } }

    // MARK: 「不相关」黑名单(该查询词下不再出现的标记)

    func blockRef(term: String, refKey: String) throws { try serialized {
        guard let db else { throw MSError("数据库未打开") }
        let stmt = try prepare("INSERT OR REPLACE INTO query_block(term, ref_key, created_at) VALUES(?,?,?)")
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_text(stmt, 1, term, -1, Self.transient)
        sqlite3_bind_text(stmt, 2, refKey, -1, Self.transient)
        sqlite3_bind_double(stmt, 3, Date().timeIntervalSince1970)
        guard sqlite3_step(stmt) == SQLITE_DONE else {
            throw MSError("黑名单写入失败: \(String(cString: sqlite3_errmsg(db)))")
        }
    } }

    func unblockRef(term: String, refKey: String) throws { try serialized {
        guard let db else { throw MSError("数据库未打开") }
        let stmt = try prepare("DELETE FROM query_block WHERE term = ? AND ref_key = ?")
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_text(stmt, 1, term, -1, Self.transient)
        sqlite3_bind_text(stmt, 2, refKey, -1, Self.transient)
        guard sqlite3_step(stmt) == SQLITE_DONE else {
            throw MSError("黑名单移除失败: \(String(cString: sqlite3_errmsg(db)))")
        }
    } }

    func blockedRefs(term: String) throws -> Set<String> { try serialized {
        guard let db, !term.isEmpty else { return [] }
        let stmt = try prepare("SELECT ref_key FROM query_block WHERE term = ?")
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_text(stmt, 1, term, -1, Self.transient)
        var out: Set<String> = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            out.insert(String(cString: sqlite3_column_text(stmt, 0)))
        }
        return out
    } }

    /// 库内标签词典:全部照片/视频帧标签及出现次数(降序)——「图像模型能识别哪些英文词」的精确答案
    /// 视频按帧计(同一标签出现在 3 帧=3 次),照片每张 1 行
    func labelVocabCounts() throws -> [(label: String, count: Int)] { try serialized {
        guard let db else { return [] }
        let stmt = try prepare("""
            SELECT title FROM items
            WHERE kind IN ('photoLabel','videoFrameLabel') AND title IS NOT NULL
            """)
        defer { sqlite3_finalize(stmt) }
        var counts: [String: Int] = [:]
        while sqlite3_step(stmt) == SQLITE_ROW {
            // 每行 title 是该照片/帧的标签串("food, fried_egg, ..."),拆开累计
            let title = String(cString: sqlite3_column_text(stmt, 0))
            for label in title.components(separatedBy: ", ") where !label.isEmpty {
                counts[label, default: 0] += 1
            }
        }
        return counts.sorted { $0.value > $1.value }.map { (label: $0.key, count: $0.value) }
    } }

    /// 某索引对象已记录的主色调(任取一行)
    func colorForRef(_ refKey: String) -> String? { try serialized {
        guard let db, let stmt = try? prepare("SELECT color FROM items WHERE ref_key = ? AND color IS NOT NULL LIMIT 1") else { return nil }
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_text(stmt, 1, refKey, -1, Self.transient)
        guard sqlite3_step(stmt) == SQLITE_ROW, sqlite3_column_text(stmt, 0) != nil else { return nil }
        return String(cString: sqlite3_column_text(stmt, 0))
    } }

    /// 已索引照片数(空态提示用)
    func countPhotos() -> Int { try serialized {
        guard let db, let stmt = try? prepare("SELECT COUNT(*) FROM items WHERE kind = 'photo'") else { return 0 }
        defer { sqlite3_finalize(stmt) }
        guard sqlite3_step(stmt) == SQLITE_ROW else { return 0 }
        return Int(sqlite3_column_int64(stmt, 0))
    } }

    /// 全部去重标签字符串(语义路由的封闭词表)
    func allDistinctLabels() throws -> [String] { try serialized {
        guard let db else { return [] }
        let stmt = try prepare("SELECT DISTINCT title FROM items WHERE kind = 'photoLabel' AND title IS NOT NULL")
        defer { sqlite3_finalize(stmt) }
        var out: [String] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            out.append(String(cString: sqlite3_column_text(stmt, 0)))
        }
        return out
    } }

    /// 标签词向量表(把任意中文查询路由到库内实际存在的英文标签)
    func labelVocab() throws -> [(label: String, vec: [Float])] { try serialized {
        guard let db else { return [] }
        let stmt = try prepare("SELECT label, vec FROM label_vocab")
        defer { sqlite3_finalize(stmt) }
        var out: [(String, [Float])] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            let label = String(cString: sqlite3_column_text(stmt, 0))
            let n = Int(sqlite3_column_bytes(stmt, 1))
            var vec = [Float](repeating: 0, count: n / 4)
            if let base = sqlite3_column_blob(stmt, 1) {
                memcpy(&vec, base, n)
            }
            out.append((label, vec))
        }
        return out.map { (label: $0.0, vec: $0.1) }
    } }

    func saveLabelVec(_ label: String, vec: [Float]) throws { try serialized {
        let stmt = try prepare("INSERT OR REPLACE INTO label_vocab(label, vec) VALUES(?,?)")
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_text(stmt, 1, label, -1, Self.transient)
        _ = vec.withUnsafeBufferPointer { buf in
            sqlite3_bind_blob(stmt, 2, buf.baseAddress, Int32(buf.count * 4), Self.transient)
        }
        guard sqlite3_step(stmt) == SQLITE_DONE else { throw MSError("标签词向量保存失败") }
    } }

    /// 移除某照片/视频/文件的**全部**索引行(各类型)
    func removeEverythingForRef(_ refKey: String) throws { try serialized {
        let stmt = try prepare("DELETE FROM items WHERE ref_key = ?")
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_text(stmt, 1, refKey, -1, Self.transient)
        sqlite3_step(stmt)
        let stmt2 = try prepare("DELETE FROM ocr_text WHERE ref_key = ?")
        defer { sqlite3_finalize(stmt2) }
        sqlite3_bind_text(stmt2, 1, refKey, -1, Self.transient)
        sqlite3_step(stmt2)
        let stmt3 = try prepare("DELETE FROM face_done WHERE ref_key = ?")
        defer { sqlite3_finalize(stmt3) }
        sqlite3_bind_text(stmt3, 1, refKey, -1, Self.transient)
        sqlite3_step(stmt3)
    } }

    func deleteUserTag(refKey: String, tag: String) throws { try serialized {
        let stmt = try prepare("DELETE FROM items WHERE kind = 'userTag' AND ref_key = ? AND title = ?")
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_text(stmt, 1, refKey, -1, Self.transient)
        sqlite3_bind_text(stmt, 2, tag, -1, Self.transient)
        sqlite3_step(stmt)
    } }

    /// 清空全部人名标签(人物归组「取消命名」用):标签行连向量一起删。
    /// 聚类只看 face 向量,清标签不影响分组结构,组名回到「未命名」
    func deleteAllUserTags() throws { try serialized {
        let stmt = try prepare("DELETE FROM items WHERE kind = 'userTag'")
        defer { sqlite3_finalize(stmt) }
        sqlite3_step(stmt)
    } }

    /// 保存照片的 OCR 文字(每照片一行,重复覆盖)
    func saveOCR(refKey: String, text: String) throws { try serialized {
        let stmt = try prepare("INSERT OR REPLACE INTO ocr_text(ref_key, text) VALUES(?,?)")
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_text(stmt, 1, refKey, -1, Self.transient)
        sqlite3_bind_text(stmt, 2, text, -1, Self.transient)
        guard sqlite3_step(stmt) == SQLITE_DONE else { throw MSError("OCR 保存失败") }
    } }

    /// 视频语音转写追加进该 ref 的 OCR 原文(〔语音〕独立段落):
    /// 帧 OCR 行已存在则追加不覆盖;总长封顶防极端长视频撑爆检索。
    /// 无文字视频已有 "" 占位行,追加后占位被替换,检索行为不变
    func appendSpeechText(refKey: String, text: String) throws { try serialized {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        var existing = ""
        let rd = try prepare("SELECT text FROM ocr_text WHERE ref_key = ?")
        sqlite3_bind_text(rd, 1, refKey, -1, Self.transient)
        if sqlite3_step(rd) == SQLITE_ROW, let c = sqlite3_column_text(rd, 0) {
            existing = String(cString: c)
        }
        sqlite3_finalize(rd)
        let merged = existing.isEmpty
            ? "〔语音〕\(trimmed)"
            : existing + "\n〔语音〕\(trimmed)"
        let stmt = try prepare("INSERT OR REPLACE INTO ocr_text(ref_key, text) VALUES(?,?)")
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_text(stmt, 1, refKey, -1, Self.transient)
        sqlite3_bind_text(stmt, 2, String(merged.prefix(30_000)), -1, Self.transient)
        guard sqlite3_step(stmt) == SQLITE_DONE else { throw MSError("语音文本保存失败") }
    } }

    /// OCR 文字精确检索(子串匹配,支持中文/数字/编号)
    func searchOCR(query: String) throws -> [(refKey: String, text: String)] { try serialized {
        guard let db, !query.isEmpty else { return [] }
        let escaped = query
            .replacingOccurrences(of: "%", with: "\\%")
            .replacingOccurrences(of: "_", with: "\\_")
        let stmt = try prepare("SELECT ref_key, text FROM ocr_text WHERE text LIKE ? ESCAPE '\\' LIMIT 4000")
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_text(stmt, 1, "%\(escaped)%", -1, Self.transient)
        var out: [(String, String)] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            out.append((String(cString: sqlite3_column_text(stmt, 0)),
                        String(cString: sqlite3_column_text(stmt, 1))))
        }
        return out.map { (refKey: $0.0, text: $0.1) }
    } }

    /// 文字检索(限定范围):只在给定 ref 集合里做子串匹配(结果内搜索用)
    func searchOCR(query: String, among: [String]) throws -> [String] { try serialized {
        guard let db, !query.isEmpty, !among.isEmpty else { return [] }
        let escaped = query
            .replacingOccurrences(of: "%", with: "\\%")
            .replacingOccurrences(of: "_", with: "\\_")
        var out = Set<String>()
        // IN 列表分片绑定,避免变量数上限
        for chunkStart in stride(from: 0, to: among.count, by: 400) {
            let part = Array(among[chunkStart..<min(chunkStart + 400, among.count)])
            let marks = Array(repeating: "?", count: part.count).joined(separator: ",")
            let stmt = try prepare("SELECT ref_key FROM ocr_text WHERE text LIKE ? ESCAPE '\\' AND ref_key IN (\(marks))")
            defer { sqlite3_finalize(stmt) }
            sqlite3_bind_text(stmt, 1, "%\(escaped)%", -1, Self.transient)
            for (i, id) in part.enumerated() {
                sqlite3_bind_text(stmt, Int32(i + 2), id, -1, Self.transient)
            }
            while sqlite3_step(stmt) == SQLITE_ROW {
                out.insert(String(cString: sqlite3_column_text(stmt, 0)))
            }
        }
        return Array(out)
    } }

    /// 重命名导入文件记录,并同步其可搜索的文件名
    func renameFile(id: String, newName: String) throws { try serialized {
        let stmt = try prepare("UPDATE files SET name = ? WHERE id = ?")
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_text(stmt, 1, newName, -1, Self.transient)
        sqlite3_bind_text(stmt, 2, id, -1, Self.transient)
        guard sqlite3_step(stmt) == SQLITE_DONE else { throw MSError("重命名记录失败") }
        let stmt2 = try prepare("UPDATE items SET title = ? WHERE kind = 'file' AND ref_key = ?")
        defer { sqlite3_finalize(stmt2) }
        sqlite3_bind_text(stmt2, 1, newName, -1, Self.transient)
        sqlite3_bind_text(stmt2, 2, id, -1, Self.transient)
        sqlite3_step(stmt2)
    } }

    func addFile(_ f: ImportedFile) throws {        let stmt = try prepare("""
        INSERT OR REPLACE INTO files(id, folder_id, rel_path, bookmark, name, size, added_at, indexed)
        VALUES(?,?,?,?,?,?,?,?)
        """)
        defer { sqlite3_finalize(stmt) }
        // ⚠️ 参数 1=id 必须绑定:此前漏绑,SQLite 以 NULL 主键写行,
        // allFiles 读 NULL id 时 String(cString:) 强解包=EXC_BREAKPOINT
        // (构建 68-70 导入文件夹后必崩的总根源)
        sqlite3_bind_text(stmt, 1, f.id, -1, Self.transient)
        if let fid = f.folderID { sqlite3_bind_int64(stmt, 2, fid) } else { sqlite3_bind_null(stmt, 2) }
        if let p = f.relPath { sqlite3_bind_text(stmt, 3, p, -1, Self.transient) } else { sqlite3_bind_null(stmt, 3) }
        if let b = f.bookmark { _ = b.withUnsafeBytes { sqlite3_bind_blob(stmt, 4, $0.baseAddress, Int32(b.count), Self.transient) } }
        else { sqlite3_bind_null(stmt, 4) }
        sqlite3_bind_text(stmt, 5, f.name, -1, Self.transient)
        sqlite3_bind_int64(stmt, 6, f.size)
        sqlite3_bind_double(stmt, 7, f.addedAt.timeIntervalSince1970)
        sqlite3_bind_int(stmt, 8, f.indexed ? 1 : 0)
        guard sqlite3_step(stmt) == SQLITE_DONE else { throw MSError("写入文件记录失败") }
    }

    func allFiles() throws -> [ImportedFile] { try serialized {
        guard let db else { return [] }
        let stmt = try prepare("""
        SELECT id, folder_id, rel_path, bookmark, name, size, added_at, indexed FROM files ORDER BY added_at DESC
        """)
        defer { sqlite3_finalize(stmt) }
        var out: [ImportedFile] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            // 毒行防御:漏绑 id 的旧版本写入过 id=NULL 的行,
            // 对 NULL 列做 String(cString:) 是强解包陷阱——跳过,绝不解包
            guard sqlite3_column_type(stmt, 0) != SQLITE_NULL else { continue }
            let id = String(cString: sqlite3_column_text(stmt, 0))
            let folderID: Int64? = sqlite3_column_type(stmt, 1) == SQLITE_NULL ? nil : sqlite3_column_int64(stmt, 1)
            let relPath: String? = sqlite3_column_type(stmt, 2) == SQLITE_NULL ? nil : String(cString: sqlite3_column_text(stmt, 2))
            let bm: Data? = sqlite3_column_type(stmt, 3) == SQLITE_NULL
                ? nil
                : Data(bytes: sqlite3_column_blob(stmt, 3), count: Int(sqlite3_column_bytes(stmt, 3)))
            let name = String(cString: sqlite3_column_text(stmt, 4))
            let size = sqlite3_column_int64(stmt, 5)
            let addedAt = Date(timeIntervalSince1970: sqlite3_column_double(stmt, 6))
            let indexed = sqlite3_column_int(stmt, 7) != 0
            out.append(ImportedFile(id: id, folderID: folderID, relPath: relPath, bookmark: bm,
                                    name: name, size: size, addedAt: addedAt, indexed: indexed))
        }
        return out
    } }

    func file(id: String) throws -> ImportedFile? { try serialized {
        try allFiles().first { $0.id == id }
    } }

    func setFileIndexed(id: String) throws { try serialized {
        let stmt = try prepare("UPDATE files SET indexed = 1 WHERE id = ?")
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_text(stmt, 1, id, -1, Self.transient)
        sqlite3_step(stmt)
    } }

    /// 全部已导入文件的 id(OCR 命中区分 文件/照片 用,files 表量级小)
    func fileIds() throws -> Set<String> { try serialized {
        guard let db else { return [] }
        let stmt = try prepare("SELECT id FROM files")
        defer { sqlite3_finalize(stmt) }
        var out = Set<String>()
        while sqlite3_step(stmt) == SQLITE_ROW {
            out.insert(String(cString: sqlite3_column_text(stmt, 0)))
        }
        return out
    } }

    /// 已有向量但缺 OCR 原文的照片(增量自动补全用;
    /// 无字照片有 "" 占位行,不会落在缺失集合里反复重处理)
    func photoRefKeysMissingOCR() throws -> Set<String> { try serialized {
        guard let db else { return [] }
        let stmt = try prepare("""
        SELECT ref_key FROM items
        WHERE kind = 'photo' AND space = 'clip'
          AND ref_key NOT IN (SELECT ref_key FROM ocr_text)
        """)
        defer { sqlite3_finalize(stmt) }
        var out = Set<String>()
        while sqlite3_step(stmt) == SQLITE_ROW {
            out.insert(String(cString: sqlite3_column_text(stmt, 0)))
        }
        return out
    } }

    /// 有帧向量但缺 OCR 行的视频(帧 OCR 自愈用)
    func videoRefKeysMissingOCR() throws -> Set<String> { try serialized {
        guard let db else { return [] }
        let stmt = try prepare("""
        SELECT DISTINCT ref_key FROM items
        WHERE kind IN ('videoFrame','videoFrameLabel')
          AND ref_key NOT IN (SELECT ref_key FROM ocr_text)
        """)
        defer { sqlite3_finalize(stmt) }
        var out = Set<String>()
        while sqlite3_step(stmt) == SQLITE_ROW {
            out.insert(String(cString: sqlite3_column_text(stmt, 0)))
        }
        return out
    } }

    /// 视频帧向量统计:有帧向量的视频数 + 每个视频的帧数(诊断用)
    func videoFrameStats() throws -> (videos: Int, perVideo: [String: Int]) { try serialized {
        guard let db else { return (0, [:]) }
        let stmt = try prepare("""
        SELECT ref_key, COUNT(*) FROM items WHERE kind = 'videoFrame' GROUP BY ref_key
        """)
        defer { sqlite3_finalize(stmt) }
        var per: [String: Int] = [:]
        while sqlite3_step(stmt) == SQLITE_ROW {
            per[String(cString: sqlite3_column_text(stmt, 0))] = Int(sqlite3_column_int(stmt, 1))
        }
        return (per.count, per)
    } }

    /// 删除某视频的全部帧向量与帧标签(重新抽帧前调用,防旧行残留)
    func removeFrames(refKey: String) throws { try serialized {
        guard let db else { return }
        let stmt = try prepare("DELETE FROM items WHERE kind IN ('videoFrame','videoFrameLabel') AND ref_key = ?")
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_text(stmt, 1, refKey, -1, Self.transient)
        sqlite3_step(stmt)
    } }

    // MARK: - 人脸(方案B:人脸识别归组)

    /// 标记照片已完成人脸处理(有脸无脸都记;人脸模型缺失时不记,装上后自动补)
    func markFaceDone(refKey: String) throws { try serialized {
        let stmt = try prepare("INSERT OR REPLACE INTO face_done(ref_key) VALUES(?)")
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_text(stmt, 1, refKey, -1, Self.transient)
        guard sqlite3_step(stmt) == SQLITE_DONE else { throw MSError("face_done 写入失败") }
    } }

    /// 已有照片向量但未做过人脸处理的照片(人脸补全队列;face_done 是唯一依据,
    /// 直接看 face 行会把无脸照片卷进无限重处理)
    func photoRefKeysMissingFaces() throws -> Set<String> { try serialized {
        guard let db else { return [] }
        let stmt = try prepare("""
        SELECT ref_key FROM items
        WHERE kind = 'photo' AND space = 'clip'
          AND ref_key NOT IN (SELECT ref_key FROM face_done)
        """)
        defer { sqlite3_finalize(stmt) }
        var out = Set<String>()
        while sqlite3_step(stmt) == SQLITE_ROW {
            out.insert(String(cString: sqlite3_column_text(stmt, 0)))
        }
        return out
    } }

    /// 人脸覆盖度(设置/资料库提示用):向量总行数 + 涉及照片数 + 已记账照片数
    func faceStats() -> (faces: Int, photos: Int, done: Int) { try serialized {
        guard let db, let stmt = try? prepare("""
            SELECT (SELECT COUNT(*) FROM items WHERE kind = 'face'),
                   (SELECT COUNT(DISTINCT ref_key) FROM items WHERE kind = 'face'),
                   (SELECT COUNT(*) FROM face_done)
            """) else { return (0, 0, 0) }
        defer { sqlite3_finalize(stmt) }
        guard sqlite3_step(stmt) == SQLITE_ROW else { return (0, 0, 0) }
        return (Int(sqlite3_column_int64(stmt, 0)),
                Int(sqlite3_column_int64(stmt, 1)),
                Int(sqlite3_column_int64(stmt, 2)))
    } }

    /// 全部人脸向量(归组扫描用):keys 形如 "refKey#脸序号",
    /// refs 与 keys 一一对应指向所属照片;blob 字节数校验防截断行越界(同 allPhotoVectors)。
    /// 被人工排除的照片不参与(face_block)
    func allFaceVectors() throws -> (keys: [String], refs: [String], flat: [Float], dim: Int) { try serialized {
        guard let db else { return ([], [], [], 0) }
        let stmt = try prepare("""
        SELECT ref_key, frame_index, vec, dim FROM items
        WHERE kind = 'face' AND space = 'face'
          AND ref_key NOT IN (SELECT ref_key FROM face_block)
        ORDER BY ref_key
        """)
        defer { sqlite3_finalize(stmt) }
        var keys: [String] = []
        var refs: [String] = []
        var flat: [Float] = []
        var dim = 0
        while sqlite3_step(stmt) == SQLITE_ROW {
            let d = Int(sqlite3_column_int(stmt, 3))
            guard d > 0, let blob = sqlite3_column_blob(stmt, 2) else { continue }
            guard sqlite3_column_bytes(stmt, 2) == d * 4 else { continue }
            if dim == 0 { dim = d } else if d != dim { continue }
            let ref = String(cString: sqlite3_column_text(stmt, 0))
            keys.append("\(ref)#\(sqlite3_column_int(stmt, 1))")
            refs.append(ref)
            flat.append(contentsOf: Array(UnsafeBufferPointer(
                start: blob.assumingMemoryBound(to: Float.self), count: d)))
        }
        return (keys, refs, flat, dim)
    } }

    /// 下一张照片的用户标签槽位(frame_index):同照片可挂多个人名(合照两人各归各组)
    func nextUserTagSlot(refKey: String) -> Int { try serialized {
        guard let db,
              let stmt = try? prepare("SELECT COALESCE(MAX(frame_index), -1) + 1 FROM items WHERE kind = 'userTag' AND ref_key = ?")
        else { return 0 }
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_text(stmt, 1, refKey, -1, Self.transient)
        return sqlite3_step(stmt) == SQLITE_ROW ? Int(sqlite3_column_int(stmt, 0)) : 0
    } }

    // MARK: - 人脸人工排除(人物归组手动纠错)

    /// 排除照片的全部人脸参与聚类(用户勾选「移出本组」的错误照片)。
    /// 幂等:重复排除无副作用;聚类在排除集合之外重新进行
    func blockFacePhotos(_ refKeys: [String]) throws { try serialized {
        for refKey in refKeys {
            let stmt = try prepare("INSERT OR REPLACE INTO face_block(ref_key) VALUES(?)")
            sqlite3_bind_text(stmt, 1, refKey, -1, Self.transient)
            sqlite3_step(stmt)
            sqlite3_finalize(stmt)
        }
    } }

    /// 当前被排除的照片数(设置区显示用)
    func blockedFacePhotoCount() -> Int { try serialized {
        guard let db, let stmt = try? prepare("SELECT COUNT(*) FROM face_block") else { return 0 }
        defer { sqlite3_finalize(stmt) }
        return sqlite3_step(stmt) == SQLITE_ROW ? Int(sqlite3_column_int(stmt, 0)) : 0
    } }

    /// 恢复全部被排除的照片(重新参与归组)
    func unblockAllFacePhotos() throws { try serialized {
        let stmt = try prepare("DELETE FROM face_block")
        defer { sqlite3_finalize(stmt) }
        sqlite3_step(stmt)
    } }

    /// 批量取照片的已有用户标签(人物归组展示组名用),IN 列表分片绑定
    func userTagsForRefs(_ refKeys: [String]) -> [String: [String]] { try serialized {
        var out: [String: [String]] = [:]
        guard let db, !refKeys.isEmpty else { return out }
        for chunkStart in stride(from: 0, to: refKeys.count, by: 400) {
            let part = Array(refKeys[chunkStart..<min(chunkStart + 400, refKeys.count)])
            let marks = Array(repeating: "?", count: part.count).joined(separator: ",")
            guard let stmt = try? prepare("SELECT ref_key, title FROM items WHERE kind = 'userTag' AND ref_key IN (\(marks))")
            else { continue }
            defer { sqlite3_finalize(stmt) }
            for (i, id) in part.enumerated() {
                sqlite3_bind_text(stmt, Int32(i + 1), id, -1, Self.transient)
            }
            while sqlite3_step(stmt) == SQLITE_ROW {
                if let t = sqlite3_column_text(stmt, 1) {
                    out[String(cString: sqlite3_column_text(stmt, 0)), default: []]
                        .append(String(cString: t))
                }
            }
        }
        return out
    } }

    // MARK: - 图像模型更换(so400m 等在线升级)

    /// 库内 clip 向量的维度(任取一行;空库返回 0)。
    /// 检索端对维度不匹配的行本就跳过,这里只为「换模型后清旧向量」提供判据
    func clipVectorDim() -> Int { try serialized {
        guard let db, let stmt = try? prepare("SELECT dim FROM items WHERE space = 'clip' LIMIT 1")
        else { return 0 }
        defer { sqlite3_finalize(stmt) }
        return sqlite3_step(stmt) == SQLITE_ROW ? Int(sqlite3_column_int(stmt, 0)) : 0
    } }

    /// 清空全部 clip 空间向量(更换图像模型、旧向量点积已无意义时)。
    /// OCR/标签/人脸/用户标签不受影响,照片/视频在下一轮同步全量重嵌
    func wipeClipVectors() throws { try serialized {
        try exec("DELETE FROM items WHERE space = 'clip'")
    } }

    /// 已有 clip 向量的照片/视频集合(增量同步的「已索引」口径)。
    /// 只认向量行本身,不认标签/人脸行——更换图像模型清空向量后,
    /// 仅剩标签行的照片也必须重嵌,否则换模型后一张都不会重建
    func refKeysWithClipVectors() throws -> Set<String> { try serialized {
        guard let db else { return [] }
        let stmt = try prepare("""
        SELECT DISTINCT ref_key FROM items WHERE space = 'clip' AND kind IN ('photo','videoFrame')
        """)
        defer { sqlite3_finalize(stmt) }
        var out = Set<String>()
        while sqlite3_step(stmt) == SQLITE_ROW {
            if let c = sqlite3_column_text(stmt, 0) {
                out.insert(String(cString: c))
            }
        }
        return out
    } }

    /// 清空全部人脸向量与记账(人脸模型更换、旧嵌入空间不通用时)。
    /// 照片/OCR/标签不受影响,下一轮同步走轻量补全通道全量重嵌
    func wipeFaceVectors() throws { try serialized {
        try exec("DELETE FROM items WHERE kind = 'face'")
        try exec("DELETE FROM face_done")
    } }

    /// 全部照片的 clip 向量(查重用):keys 与 flat 内的行一一对应
    func allPhotoVectors() throws -> (keys: [String], flat: [Float], dim: Int) { try serialized {
        guard let db else { return ([], [], 0) }
        let stmt = try prepare("""
        SELECT ref_key, vec, dim FROM items WHERE kind = 'photo' AND space = 'clip' ORDER BY ref_key
        """)
        defer { sqlite3_finalize(stmt) }
        var keys: [String] = []
        var flat: [Float] = []
        var dim = 0
        while sqlite3_step(stmt) == SQLITE_ROW {
            let d = Int(sqlite3_column_int(stmt, 2))
            guard d > 0, let blob = sqlite3_column_blob(stmt, 1) else { continue }
            // blob 实际字节数必须与 dim 声明一致:截断行会让 UnsafeBufferPointer
            // 越界读 sqlite 堆内存 → vDSP_mmul SIGSEGV(2026-10-11 查重崩溃实锤)
            guard sqlite3_column_bytes(stmt, 1) == d * 4 else { continue }
            if dim == 0 { dim = d } else if d != dim { continue }
            keys.append(String(cString: sqlite3_column_text(stmt, 0)))
            flat.append(contentsOf: Array(UnsafeBufferPointer(
                start: blob.assumingMemoryBound(to: Float.self), count: d)))
        }
        return (keys, flat, dim)
    } }

    func deleteFile(id: String) throws { try serialized {
        let stmt = try prepare("DELETE FROM files WHERE id = ?")
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_text(stmt, 1, id, -1, Self.transient)
        sqlite3_step(stmt)
        // 全量清除:文件名向量、每页文本向量(fileChunk)、PDF OCR 原文(ocr_text)
        try removeEverythingForRef(id)
    } }

    var fileCount: Int {
        guard let db else { return 0 }
        guard let stmt = try? prepare("SELECT COUNT(*) FROM files") else { return 0 }
        defer { sqlite3_finalize(stmt) }
        return sqlite3_step(stmt) == SQLITE_ROW ? Int(sqlite3_column_int(stmt, 0)) : 0
    }

    // MARK: - SQLite 辅助

    private func prepare(_ sql: String) throws -> OpaquePointer {
        guard let db else { throw MSError("数据库未打开") }
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK, let stmt else {
            throw MSError(String(cString: sqlite3_errmsg(db)))
        }
        return stmt
    }
}

struct MSError: LocalizedError {
    let message: String
    init(_ message: String) { self.message = message }
    var errorDescription: String? { message }
}
