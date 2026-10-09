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
    }

    func fail(_ error: Error) {
        errorMessage = error.localizedDescription
    }

    func notify(_ message: String) {
        successMessage = message
    }

    /// 给照片/视频添加人名或主题标签(生成 Gemma 语义向量)
    func addUserTag(refKey: String, tag: String) async throws {
        guard let gemma = models.gemma else { throw MSError("文本模型未就绪") }
        let trimmed = tag.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        let vec = try await Task.detached(priority: .userInitiated) {
            try gemma.embedDocument("person: \(trimmed)")
        }.value
        try store.upsert(kind: .userTag, refKey: refKey, space: gemma.space,
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
