import SwiftUI

// MARK: - 更新清单(CI 发布到 GitHub Release 的 manifest.json)

struct ModelPkgFile: Codable, Identifiable {
    let key: String      // full / face / so400m
    let name: String     // 显示名
    let url: String
    let size: Int64
    let note: String
    var id: String { key }
}

struct ModelManifest: Codable {
    let version: String
    let date: String
    let files: [ModelPkgFile]
}

// MARK: - 下载器(带进度)

/// 在线模型更新:读 Release 上的 manifest.json,下载 zip 后走 importModelZip。
/// 下载用 delegate 拿进度(模型包几百 MB,没有进度无法用);
/// didFinish 与 didCompleteWithError 互斥恢复 continuation,不会双 resume。
@MainActor
final class ModelUpdateLoader: NSObject, ObservableObject, URLSessionDownloadDelegate {
    @Published var checking = false
    @Published var manifest: ModelManifest?
    @Published var errorText: String?
    @Published var downloadingKey: String?
    @Published var progress: Double = 0
    @Published var importing = false

    /// 仓库公开 Release,固定 tag=models;App 严禁公开发布,此地址仅本人设备使用
    static let manifestURL =
        "https://github.com/yangru158158-ux/MediaSeek/releases/download/models/manifest.json"

    private var continuation: CheckedContinuation<URL, Error>?

    func check() async {
        checking = true
        errorText = nil
        defer { checking = false }
        do {
            var req = URLRequest(url: URL(string: Self.manifestURL)!)
            req.timeoutInterval = 30
            let (data, resp) = try await URLSession.shared.data(for: req)
            if let http = resp as? HTTPURLResponse, http.statusCode != 200 {
                throw MSError("服务器返回 \(http.statusCode)")
            }
            manifest = try JSONDecoder().decode(ModelManifest.self, from: data)
        } catch {
            errorText = "获取更新信息失败:\(error.localizedDescription)\nGitHub 直连在部分网络下不稳,可稍后重试或改用 zip 手动导入。"
        }
    }

    func download(_ file: ModelPkgFile) async throws -> URL {
        downloadingKey = file.key
        progress = 0
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForResource = 7200   // 大包慢网也要能下完
        let session = URLSession(configuration: config, delegate: self, delegateQueue: nil)
        defer { session.finishTasksAndInvalidate() }
        return try await withCheckedThrowingContinuation { c in
            self.continuation = c
            session.downloadTask(with: URL(string: file.url)!).resume()
        }
    }

    func finish(key: String, version: String) {
        UserDefaults.standard.set(version, forKey: "modelPkgVersion")
        UserDefaults.standard.set(version, forKey: "pkg_\(key)")
        downloadingKey = nil
        progress = 0
    }

    func abortWith(error: Error) {
        downloadingKey = nil
        errorText = "下载/导入失败:\(error.localizedDescription)"
    }

    // MARK: URLSessionDownloadDelegate

    nonisolated func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                                didWriteData bytesWritten: Int64, totalBytesWritten: Int64,
                                totalBytesExpectedToWrite: Int64) {
        let p = totalBytesExpectedToWrite > 0
            ? Double(totalBytesWritten) / Double(totalBytesExpectedToWrite) : 0
        Task { @MainActor in progress = p }
    }

    nonisolated func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                                didFinishDownloadingTo location: URL) {
        let dest = FileManager.default.temporaryDirectory
            .appendingPathComponent("model-pkg-\(UUID().uuidString).zip")
        try? FileManager.default.removeItem(at: dest)
        do {
            try FileManager.default.moveItem(at: location, to: dest)
            let c = continuation
            continuation = nil
            c?.resume(returning: dest)
        } catch {
            let c = continuation
            continuation = nil
            c?.resume(throwing: error)
        }
    }

    nonisolated func urlSession(_ session: URLSession, task: URLSessionTask,
                                didCompleteWithError error: Error?) {
        guard let error else { return }   // 正常结束时 didFinish 已恢复
        let c = continuation
        continuation = nil
        c?.resume(throwing: error)
    }
}

// MARK: - 设置页「模型在线更新」区

struct ModelUpdateSection: View {
    @EnvironmentObject var app: AppModel
    @StateObject private var loader = ModelUpdateLoader()

    private func fmtSize(_ bytes: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
    }

    var body: some View {
        Section {
            LabeledContent("已装版本",
                           value: UserDefaults.standard.string(forKey: "modelPkgVersion") ?? "手动导入/未知")
            Button {
                Task { await loader.check() }
            } label: {
                if loader.checking {
                    HStack { ProgressView(); Text("检查中…") }
                } else {
                    Label("检查更新", systemImage: "arrow.triangle.2.circlepath")
                }
            }
            .disabled(loader.checking || loader.downloadingKey != nil)

            if let m = loader.manifest {
                Text("服务端版本 \(m.version)(\(m.date))")
                    .font(.caption).foregroundStyle(.secondary)
                ForEach(m.files) { f in
                    pkgRow(f, manifestVersion: m.version)
                }
            }
            if let e = loader.errorText {
                Text(e).font(.caption2).foregroundStyle(.orange)
            }
        } header: {
            Text("模型在线更新")
        } footer: {
            Text("从 GitHub Release 下载模型包并自动导入,建议 Wi-Fi 且保持 App 前台。增强图像模型(so400m)识别更强,导入后需点「同步新增内容」全量重建图像向量(约 1-2 小时),期间可随时中断续跑。")
        }
    }

    @ViewBuilder
    private func pkgRow(_ f: ModelPkgFile, manifestVersion: String) -> some View {
        let isDownloading = loader.downloadingKey == f.key
        let installed = UserDefaults.standard.string(forKey: "pkg_\(f.key)")
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(f.name).font(.subheadline)
                    Text("\(fmtSize(f.size)) · \(f.note)")
                        .font(.caption2).foregroundStyle(.secondary)
                }
                Spacer()
                if isDownloading {
                    ProgressView()
                } else if loader.downloadingKey == nil {
                    Button(installed == manifestVersion ? "重装" : "下载") {
                        install(f, version: manifestVersion)
                    }
                    .font(.subheadline)
                }
            }
            if isDownloading {
                ProgressView(value: loader.progress)
                Text("下载 \(Int(loader.progress * 100))% — 请保持 App 在前台")
                    .font(.caption2).foregroundStyle(.secondary)
            }
            if loader.importing, isDownloading {
                Text("正在导入(首次编译模型需要一些时间)…")
                    .font(.caption2).foregroundStyle(.secondary)
            }
        }
        .disabled(loader.downloadingKey != nil && !isDownloading)
    }

    private func install(_ f: ModelPkgFile, version: String) {
        Task {
            do {
                let zipURL = try await loader.download(f)
                loader.importing = true
                try await app.models.importModelZip(at: zipURL)
                try? FileManager.default.removeItem(at: zipURL)
                loader.importing = false
                loader.finish(key: f.key, version: version)
                app.notify("「\(f.name)」已导入并生效")
            } catch {
                loader.importing = false
                loader.abortWith(error: error)
            }
        }
    }
}
