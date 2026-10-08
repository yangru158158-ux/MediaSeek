import SwiftUI

struct RootView: View {
    @EnvironmentObject var app: AppModel

    var body: some View {
        TabView {
            SearchView()
                .tabItem { Label("搜索", systemImage: "magnifyingglass") }
            LibraryView()
                .tabItem { Label("资料库", systemImage: "photo.on.rectangle.angled") }
            SettingsView()
                .tabItem { Label("设置", systemImage: "gearshape") }
        }
        .task { await app.bootstrap() }
        .alert("出错了", isPresented: .init(
            get: { app.errorMessage != nil },
            set: { if !$0 { app.errorMessage = nil } })) {
            Button("好") { app.errorMessage = nil }
        } message: {
            Text(app.errorMessage ?? "")
        }
    }
}
