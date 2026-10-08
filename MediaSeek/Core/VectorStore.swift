import Foundation
import SQLite3
import Accelerate

/// 媒体条目类别。photoLabel 是照片的 Vision 标签文本向量(gemma 空间),
/// 让"一只猫"这类中文查询通过多语言文本模型命中照片。
enum ItemKind: String, CaseIterable {
    case photo, photoLabel, videoFrame, file, fileChunk
}

struct SearchHit {
    let kind: ItemKind
    let refKey: String
    let frameIndex: Int
    let space: String
    let title: String?
    let date: Date?
    let score: Float
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

    // MARK: - 打开 / 建表

    func open() throws {
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
          UNIQUE(kind, ref_key, frame_index)
        )
        """)
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
    }

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

    // MARK: - 向量条目

    func upsert(kind: ItemKind, refKey: String, frameIndex: Int = 0,
                space: String, vector: [Float], title: String?, date: Date?) throws {
        guard let db else { throw MSError("数据库未打开") }
        let sql = """
        INSERT INTO items(kind, ref_key, frame_index, space, dim, vec, title, created_at)
        VALUES(?,?,?,?,?,?,?,?)
        ON CONFLICT(kind, ref_key, frame_index) DO UPDATE SET
          space=excluded.space, dim=excluded.dim, vec=excluded.vec,
          title=excluded.title, created_at=excluded.created_at
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
            guard sqlite3_step(stmt) == SQLITE_DONE else {
                throw MSError(String(cString: sqlite3_errmsg(db)))
            }
        }
    }

    func refKeys(kinds: [ItemKind]) throws -> Set<String> {
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
    }

    /// 库同步:删除已不在系统相册里的条目
    func removeRefs(kinds: [ItemKind], notIn keep: Set<String>) throws {
        guard let db else { return }
        let existing = try refKeys(kinds: kinds)
        let stale = existing.subtracting(keep)
        guard !stale.isEmpty else { return }
        let ph = kinds.map { _ in "?" }.joined(separator: ",")
        let stmt = try prepare("DELETE FROM items WHERE kind IN (\(ph)) AND ref_key = ?")
        defer { sqlite3_finalize(stmt) }
        try exec("BEGIN")
        for key in stale {
            for (i, k) in kinds.enumerated() {
                sqlite3_bind_text(stmt, Int32(i + 1), k.rawValue, -1, Self.transient)
            }
            sqlite3_bind_text(stmt, Int32(kinds.count + 1), key, -1, Self.transient)
            sqlite3_step(stmt)
            sqlite3_reset(stmt)
            sqlite3_clear_bindings(stmt)
        }
        try exec("COMMIT")
    }

    func removeByRef(kind: ItemKind, refKey: String) throws {
        let stmt = try prepare("DELETE FROM items WHERE kind = ? AND ref_key = ?")
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_text(stmt, 1, kind.rawValue, -1, Self.transient)
        sqlite3_bind_text(stmt, 2, refKey, -1, Self.transient)
        sqlite3_step(stmt)
    }

    func counts() throws -> [ItemKind: Int] {
        guard let db else { return [:] }
        let stmt = try prepare("""
        SELECT kind, COUNT(DISTINCT ref_key) FROM items
        WHERE kind != 'photoLabel' GROUP BY kind
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
    }

    func deleteAll() throws {
        try exec("DELETE FROM items")
        try exec("UPDATE files SET indexed = 0")
    }

    /// 余弦检索(向量写入时已归一化,点积即余弦)
    func search(space: String, kinds: [ItemKind], query: [Float],
                limit: Int, minScore: Float) throws -> [SearchHit] {
        guard let db else { return [] }
        let ph = kinds.map { _ in "?" }.joined(separator: ",")
        let stmt = try prepare("""
        SELECT kind, ref_key, frame_index, title, created_at, dim, vec
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
            let frameIndex = Int(sqlite3_column_int(stmt, 2))
            let title = sqlite3_column_type(stmt, 3) == SQLITE_NULL
                ? nil : String(cString: sqlite3_column_text(stmt, 3))
            let ts = sqlite3_column_double(stmt, 4)
            let date = ts > 0 ? Date(timeIntervalSince1970: ts) : nil
            let dim = Int(sqlite3_column_int(stmt, 5))
            guard dim == q.count, let blob = sqlite3_column_blob(stmt, 6) else { continue }
            let v = blob.assumingMemoryBound(to: Float.self)
            let score = vDSP.dot(Array(UnsafeBufferPointer(start: v, count: dim)), q)
            if score >= minScore {
                hits.append(SearchHit(kind: kind, refKey: refKey, frameIndex: frameIndex,
                                      space: space, title: title, date: date, score: score))
            }
        }
        return Array(hits.sorted { $0.score > $1.score }.prefix(limit))
    }

    // MARK: - 导入文件

    func addFolder(bookmark: Data) throws -> Int64 {
        let stmt = try prepare("INSERT INTO folders(bookmark, added_at) VALUES(?,?)")
        defer { sqlite3_finalize(stmt) }
        _ = bookmark.withUnsafeBytes { sqlite3_bind_blob(stmt, 1, $0.baseAddress, Int32(bookmark.count), Self.transient) }
        sqlite3_bind_double(stmt, 2, Date().timeIntervalSince1970)
        guard sqlite3_step(stmt) == SQLITE_DONE else { throw MSError("写入文件夹记录失败") }
        return sqlite3_last_insert_rowid(db)
    }

    func folderBookmark(id: Int64) throws -> Data {
        let stmt = try prepare("SELECT bookmark FROM folders WHERE id = ?")
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_int64(stmt, 1, id)
        guard sqlite3_step(stmt) == SQLITE_ROW,
              let blob = sqlite3_column_blob(stmt, 0) else {
            throw MSError("找不到源文件夹,可能已被移动或删除")
        }
        return Data(bytes: blob, count: Int(sqlite3_column_bytes(stmt, 0)))
    }

    func addFile(_ f: ImportedFile) throws {        let stmt = try prepare("""
        INSERT OR REPLACE INTO files(id, folder_id, rel_path, bookmark, name, size, added_at, indexed)
        VALUES(?,?,?,?,?,?,?,?)
        """)
        defer { sqlite3_finalize(stmt) }
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

    func allFiles() throws -> [ImportedFile] {
        guard let db else { return [] }
        let stmt = try prepare("""
        SELECT id, folder_id, rel_path, bookmark, name, size, added_at, indexed FROM files ORDER BY added_at DESC
        """)
        defer { sqlite3_finalize(stmt) }
        var out: [ImportedFile] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
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
    }

    func file(id: String) throws -> ImportedFile? {
        try allFiles().first { $0.id == id }
    }

    func setFileIndexed(id: String) throws {
        let stmt = try prepare("UPDATE files SET indexed = 1 WHERE id = ?")
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_text(stmt, 1, id, -1, Self.transient)
        sqlite3_step(stmt)
    }

    func deleteFile(id: String) throws {
        let stmt = try prepare("DELETE FROM files WHERE id = ?")
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_text(stmt, 1, id, -1, Self.transient)
        sqlite3_step(stmt)
        try removeByRef(kind: .file, refKey: id)
        try removeByRef(kind: .fileChunk, refKey: id)
    }

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
