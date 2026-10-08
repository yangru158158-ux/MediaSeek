import SwiftUI
import Photos

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

    init() {
        let s = VectorStore()
        store = s
        imports = ImportLibrary(store: s)
        let m = ModelManager()
        models = m
        indexing = IndexingCoordinator(store: s, models: m, photo: photoLib, imports: imports)
        search = SearchEngine(store: s, models: m, photo: photoLib, imports: imports)
    }

    func fail(_ error: Error) {
        errorMessage = error.localizedDescription
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
