import SwiftUI
import Photos
import Combine

@main
struct MediaSeekApp: App {
    @StateObject private var app = AppModel()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(app)
        }
    }
}

/// 全局依赖容器:数据库、模型管理、索引编排、检索引擎
@MainActor
final class AppModel: ObservableObject {
    let store: VectorStore
    let photoLib = PhotoLibraryService()
    let imports: ImportLibrary
    let models: ModelManager
    let indexing: IndexingCoordinator
    let search: SearchEngine

    @Published var errorMessage: String?
    @Published var successMessage: String?
    @Published var recentSearches: [String] = UserDefaults.standard.stringArray(forKey: "recentSearches") ?? []
    private var bag = Set<AnyCancellable>()

    init() {
        let s = VectorStore()
        store = s
        imports = ImportLibrary(store: s)
        let m = ModelManager()
        models = m
        indexing = IndexingCoordinator(store: s, models: m, photo: photoLib, imports: imports)
        search = SearchEngine(store: s, models: m, photo: photoLib, imports: imports)

        // 子服务都是 ObservableObject,但视图只观察 AppModel;
        // 把它们的状态变化统一转发,否则授权/模型/索引进度界面不会刷新
        photoLib.objectWillChange.sink { [weak self] _ in self?.objectWillChange.send() }.store(in: &bag)
        models.objectWillChange.sink { [weak self] _ in self?.objectWillChange.send() }.store(in: &bag)
        indexing.objectWillChange.sink { [weak self] _ in self?.objectWillChange.send() }.store(in: &bag)
        imports.objectWillChange.sink { [weak self] _ in self?.objectWillChange.send() }.store(in: &bag)

        // 人脸模型指纹变化(导入/在线升级)→ 立即增量同步:同步开始时自动清空
        // 旧模型脸向量并走人脸补全重嵌。此前只在 App 启动时自动同步,导入完
        // 不重启就没人触发,用户拿着旧向量去归组只能看到混人
        NotificationCenter.default.publisher(for: .faceModelChanged)
            .sink { [weak self] _ in
                Task { @MainActor in self?.indexing.startAutoSyncIfAuthorized() }
            }
            .store(in: &bag)
    }

    func fail(_ error: Error) {
        errorMessage = error.localizedDescription
    }

    func notify(_ message: String) {
        successMessage = message
    }

    /// 记录一次搜索(最新在前,去重,最多保留 20 条)
    func addSearchHistory(_ query: String) {
        let t = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !t.isEmpty else { return }
        var list = recentSearches.filter { $0 != t }
        list.insert(t, at: 0)
        recentSearches = Array(list.prefix(20))
        UserDefaults.standard.set(recentSearches, forKey: "recentSearches")
    }

    func removeSearchHistory(_ query: String) {
        recentSearches.removeAll { $0 == query }
        UserDefaults.standard.set(recentSearches, forKey: "recentSearches")
    }

    func clearSearchHistory() {
        recentSearches.removeAll()
        UserDefaults.standard.removeObject(forKey: "recentSearches")
    }

    /// 给照片/视频添加人名或主题标签(生成 Gemma 语义向量)。
    /// frame_index 用递增槽位:同一张合照可以挂多个人名(人物归组分别命名);
    /// 已有同名标签则跳过,不重复建行
    func addUserTag(refKey: String, tag: String) async throws {
        guard let gemma = models.gemma else { throw MSError("文本模型未就绪") }
        let trimmed = tag.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        if (try? store.userTags(forRefKey: refKey))?.contains(trimmed) == true { return }
        let vec = try await Task.detached(priority: .userInitiated) {
            try gemma.embedDocument("person: \(trimmed)")
        }.value
        let slot = store.nextUserTagSlot(refKey: refKey)
        try store.upsert(kind: .userTag, refKey: refKey, frameIndex: slot, space: gemma.space,
                         vector: vec, title: trimmed, date: Date())
    }

    func bootstrap() async {
        do {
            try store.open()
        } catch {
            fail(error)
        }
        await models.loadAll()
        indexing.startAutoSyncIfAuthorized()
    }
}
