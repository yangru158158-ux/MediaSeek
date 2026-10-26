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

// MARK: - 下载器(断点续传)

/// 在线模型更新:读 Release 上的 manifest.json,下载 zip 后走 importModelZip。
/// 第七十六批:下载改为「Range 断点续传 + .part 落盘」——旧实现是前台
/// 一次性 downloadTask,切后台/锁屏/断网即取消且已下数据全丢,2.1GB 的
/// so400m 包重下一次就是一次赌运气(装机实测点名)。现在:
/// - 分片写入 Application Support/MediaSeek/Downloads/pkg_{key}.part
/// - 再点下载带 Range: bytes=已下- 续传(GitHub Release 存储支持 206)
/// - 服务器返回 200(Range 被忽略/无 .part)自动从零;包已更新(.part 比
///   清单大)丢弃重下;下完核对总字节数与清单一致才进导入
/// - 进度从「已下部分/总大小」起步,界面标「续传中」
/// @MainActor 收状态;完成校验以磁盘上的 .part 实际大小为准(唯一真相,
/// 回调 Task 乱序时按盘上字节数判不完整→保留续传,自愈——验收 low)
@MainActor
final class ModelUpdateLoader: NSObject, ObservableObject, URLSessionDataDelegate {
    @Published var checking = false
    @Published var manifest: ModelManifest?
    @Published var errorText: String?
    @Published var downloadingKey: String?
    @Published var progress: Double = 0
    @Published var importing = false
    /// 本次是否为续传(进度从已下部分起步)
    @Published var resuming = false

    /// 仓库公开 Release,固定 tag=models;App 严禁公开发布,此地址仅本人设备使用
    static let manifestURL =
        "https://github.com/yangru158158-ux/MediaSeek/releases/download/models/manifest.json"

    private var continuation: CheckedContinuation<URL, Error>?
    private var dataTask: URLSessionDataTask?
    private var fileHandle: FileHandle?
    private var partPath: String = ""
    private var existingBytes: Int64 = 0
    private var received: Int64 = 0
    private var totalBytes: Int64 = 0
    private var partIsValid = false

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
        resuming = false
        let dir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("MediaSeek/Downloads", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let part = dir.appendingPathComponent("pkg_\(file.key).part")
        var existing = Int64((try? FileManager.default.attributesOfItem(atPath: part.path))?[.size] as? Int64 ?? 0)
        // 服务器包已更新:.part 比清单标称还大 = 旧包残片,丢弃重下
        if existing > file.size {
            try? FileManager.default.removeItem(at: part)
            existing = 0
        }
        // .part 已与清单等大:上次已下完只是校验/搬移没跑完(末尾被杀等)
        // ——直接消费交货。不能再发 Range(bytes=size- 不可满足,服务器回
        // 416 落进通用报错,完整文件永远无人消费=该包永久卡死(验收 medium)
        if existing == file.size, existing > 0 {
            let dest = FileManager.default.temporaryDirectory
                .appendingPathComponent("model-pkg-\(UUID().uuidString).zip")
            try? FileManager.default.removeItem(at: dest)
            try FileManager.default.moveItem(atPath: part.path, toPath: dest.path)
            progress = 1
            resuming = false
            downloadingKey = nil
            return dest
        }
        var req = URLRequest(url: URL(string: file.url)!)
        req.timeoutInterval = 60
        if existing > 0 {
            req.setValue("bytes=\(existing)-", forHTTPHeaderField: "Range")
        }
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForResource = 7200   // 大包慢网也要能下完
        let session = URLSession(configuration: config, delegate: self, delegateQueue: nil)
        defer { session.finishTasksAndInvalidate() }
        partPath = part.path
        existingBytes = existing
        received = 0
        totalBytes = file.size
        partIsValid = false
        resuming = existing > 0
        progress = existing > 0 ? Double(existing) / Double(file.size) : 0
        return try await withCheckedThrowingContinuation { c in
            self.continuation = c
            self.dataTask = session.dataTask(with: req)
            self.dataTask?.resume()
        }
    }

    func finish(key: String, version: String) {
        UserDefaults.standard.set(version, forKey: "modelPkgVersion")
        UserDefaults.standard.set(version, forKey: "pkg_\(key)")
        downloadingKey = nil
        progress = 0
        resuming = false
    }

    func abortWith(error: Error) {
        downloadingKey = nil
        errorText = "下载/导入失败:\(error.localizedDescription)"
        resuming = false
    }

    private func finishTask(_ result: Result<URL, Error>) {
        fileHandle?.closeFile()
        fileHandle = nil
        dataTask = nil
        guard let c = continuation else { return }
        continuation = nil
        switch result {
        case .success(let url): c.resume(returning: url)
        case .failure(let e): c.resume(throwing: e)
        }
    }

    // MARK: URLSessionDataDelegate

    nonisolated func urlSession(_ session: URLSession, dataTask: URLSessionDataTask,
                                didReceive response: URLResponse,
                                completionHandler: @escaping (URLSession.ResponseDisposition) -> Void) {
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        Task { @MainActor in
            let fm = FileManager.default
            if status == 206, self.existingBytes > 0 {
                // 续传:追加写(.part 必在——existingBytes>0 即以其存在为前提)
                if !fm.fileExists(atPath: self.partPath) {
                    fm.createFile(atPath: self.partPath, contents: nil)
                    self.existingBytes = 0
                }
                do {
                    self.fileHandle = try FileHandle(forWritingTo: URL(fileURLWithPath: self.partPath))
                    try self.fileHandle?.seekToEnd()
                    self.partIsValid = true
                    completionHandler(.allow)
                } catch {
                    self.finishTask(.failure(MSError("打开续传文件失败:\(error.localizedDescription)")))
                    completionHandler(.cancel)
                }
            } else if status == 200 {
                // Range 被忽略或全新下载:整包从零(丢弃已有 .part)
                try? fm.removeItem(atPath: self.partPath)
                fm.createFile(atPath: self.partPath, contents: nil)
                do {
                    self.fileHandle = try FileHandle(forWritingTo: URL(fileURLWithPath: self.partPath))
                    self.existingBytes = 0
                    self.resuming = false
                    self.progress = 0
                    self.partIsValid = true
                    completionHandler(.allow)
                } catch {
                    self.finishTask(.failure(MSError("创建下载文件失败:\(error.localizedDescription)")))
                    completionHandler(.cancel)
                }
            } else {
                self.finishTask(.failure(MSError("服务器返回 \(status),稍后重试")))
                completionHandler(.cancel)
            }
        }
    }

    nonisolated func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        let chunk = data
        Task { @MainActor in
            guard let handle = self.fileHandle, self.partIsValid else { return }
            // throwing 写(批76二修):旧 write(_:) 非 throwing,磁盘满抛 ObjC
            // 异常无法捕获直接崩进程(验收 high);失败走取消+报错,保留 .part
            do {
                try handle.write(contentsOf: chunk)
                self.received += Int64(chunk.count)
                if self.totalBytes > 0 {
                    self.progress = Double(self.existingBytes + self.received) / Double(self.totalBytes)
                }
            } catch {
                self.finishTask(.failure(MSError("写入失败(磁盘满?):\(error.localizedDescription)")))
                self.dataTask?.cancel()
            }
        }
    }

    nonisolated func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        Task { @MainActor in
            guard let error else {
                // 成功:校验以磁盘上的 .part 实际大小为准(唯一真相;回调
                // Task 乱序时计数可能滞后,按盘上字节判不完整→保留续传自愈)
                let onDisk = Int64((try? FileManager.default.attributesOfItem(atPath: self.partPath))?[.size] as? Int64 ?? 0)
                guard self.partIsValid, self.totalBytes > 0, onDisk == self.totalBytes else {
                    // 不完整:保留 .part 供续传,报错让用户重点
                    self.finishTask(.failure(MSError("下载不完整(\(onDisk)/\(self.totalBytes) 字节),再点一次从断点续传")))
                    return
                }
                let dest = FileManager.default.temporaryDirectory
                    .appendingPathComponent("model-pkg-\(UUID().uuidString).zip")
                do {
                    try? FileManager.default.removeItem(at: dest)
                    try FileManager.default.moveItem(atPath: self.partPath, toPath: dest.path)
                    self.finishTask(.success(dest))
                } catch {
                    self.finishTask(.failure(error))
                }
                return
            }
            // 失败/取消:保留 .part(断点资产),下次续传
            self.finishTask(.failure(error))
        }
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
            Text("从 GitHub Release 下载模型包并自动导入,建议 Wi-Fi 且保持 App 前台。增强图像模型(so400m)识别更强,导入后自动切换并全量重建视觉向量(约 1-2 小时,可中断续跑);之后可在上方「图像模型切换」随时切回标准模型对比效果。")
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
                Text(loader.resuming
                     ? "续传中 \(Int(loader.progress * 100))%(已从断点继续)— 中断不怕,再点从断点续"
                     : "下载 \(Int(loader.progress * 100))% — 中断不怕,已下部分自动保留续传")
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
                // so400m 增强包落独立槽位 ImageEmbedderPro(第四十六批):
                // 与随包标准模型并存,导入后自动切到 so400m,设置页可随时切回
                let isPro = f.key == "so400m"
                try await app.models.importModelZip(at: zipURL, imageTarget: isPro ? .imagePro : nil)
                try? FileManager.default.removeItem(at: zipURL)
                loader.importing = false
                loader.finish(key: f.key, version: version)
                if isPro {
                    await app.models.setImageModel(pro: true)
                    app.notify("so400m 已导入并切换;视觉向量开始全量重嵌(约 1-2 小时,可中断续跑)")
                } else {
                    app.notify("「\(f.name)」已导入并生效")
                }
            } catch {
                loader.importing = false
                loader.abortWith(error: error)
            }
        }
    }
}
