import Foundation
import AVFoundation
import Speech

/// 视频语音转文字:AVAsset 抽音轨(WAV PCM 16kHz)→ SFSpeechRecognizer 识别。
/// 第三十八批:m4a 导出在真机上可能产出坏文件(StackOverflow 实锤,模拟器
/// 正常真机无声),换成 AVAssetReader+Writer 直写 WAV PCM——Speech 框架
/// 最可靠的输入格式。识别文本追加进 ocr_text(〔语音〕段落),被 文字: 检索。
enum SpeechTranscriber {
    static let noSpeechMarker = "(未识别到语音)"
    static let maxAudioSeconds: Double = 1800

    static func requestAuthorization() async -> Bool {
        await withCheckedContinuation { cont in
            SFSpeechRecognizer.requestAuthorization { status in
                cont.resume(returning: status == .authorized)
            }
        }
    }

    static func diagnosticInfo() -> String {
        guard let rec = SFSpeechRecognizer(locale: Locale(identifier: "zh-CN")) else {
            return "zh-CN 识别器创建失败(系统不支持该语言)"
        }
        let onDevice = rec.supportsOnDeviceRecognition
        let available = rec.isAvailable
        return "zh-CN 识别器:可用=\(available), 本机模型=\(onDevice ? "已支持" : "未下载")"
    }

    static func onDeviceSupported() -> Bool {
        guard let rec = SFSpeechRecognizer(locale: Locale(identifier: "zh-CN")) else { return false }
        return rec.supportsOnDeviceRecognition
    }

    /// 转写一部视频的音轨:抽 WAV → SFSpeechRecognizer
    static func transcribeVideo(avAsset: AVAsset) async throws -> String {
        let duration = try await avAsset.load(.duration)
        guard duration.seconds > 0.5 else { return "" }

        // WAV PCM 16kHz 单声道(第三十八批):m4a 在真机上可能坏,PCM 直写
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("speech-\(UUID().uuidString).wav")
        let maxSec = min(duration.seconds, maxAudioSeconds)
        try await exportWAV(from: avAsset, to: tmp, maxSeconds: maxSec)
        defer { try? FileManager.default.removeItem(at: tmp) }

        guard let recognizer = SFSpeechRecognizer(locale: Locale(identifier: "zh-CN")),
              recognizer.isAvailable else {
            throw MSError("语音识别服务不可用")
        }
        let request = SFSpeechURLRecognitionRequest(url: tmp)
        request.shouldReportPartialResults = true
        let allowServer = UserDefaults.standard.object(forKey: "speechAllowServer") as? Bool ?? true
        request.requiresOnDeviceRecognition = !allowServer

        return try await withCheckedThrowingContinuation { cont in
            let gate = ResumeGate(cont)
            let task = recognizer.recognitionTask(with: request) { result, error in
                if let result {
                    gate.absorb(result.bestTranscription.formattedString)
                    if result.isFinal { gate.finish(error: error); return }
                }
                if let error { gate.finish(error: error); return }
            }
            gate.attach(task)
        }
    }

    /// 手写 WAV 文件(第三十八批):AVAssetWriter 不支持 WAV+PCM 的
    /// outputSettings(实测 init 即抛 NSInvalidArgumentException→SIGABRT,
    /// .ips 定案),改为 AVAssetReader 读 PCM → 手写 44 字节 WAV 头 +
    /// FileHandle 逐块写原始数据。WAV 是最简音频容器,手写比框架更可靠。
    private static func exportWAV(from avAsset: AVAsset, to url: URL, maxSeconds: Double) async throws {
        guard let audioTrack = try await avAsset.loadTracks(withMediaType: .audio).first else {
            throw MSError("无音轨")
        }
        // PCM 16kHz 单声道 16bit:Speech 框架推荐格式
        let sampleRate = 16000.0
        let channels = 1
        let bitsPerSample = 16
        let byteRate = Int(sampleRate) * channels * bitsPerSample / 8
        let blockAlign = channels * bitsPerSample / 8

        let reader = try AVAssetReader(asset: avAsset)
        let readerOut = AVAssetReaderTrackOutput(track: audioTrack, outputSettings: [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: sampleRate,
            AVNumberOfChannelsKey: channels,
            AVLinearPCMBitDepthKey: bitsPerSample,
            AVLinearPCMIsFloatKey: false,
            AVLinearPCMIsBigEndianKey: false,
        ])
        reader.add(readerOut)
        reader.timeRange = CMTimeRange(
            start: .zero,
            duration: CMTime(seconds: maxSeconds, preferredTimescale: 600))
        guard reader.startReading() else {
            throw reader.error ?? MSError("音轨读取启动失败")
        }

        // 先写 44 字节标准 WAV 头(占位 dataSize=0,最后回填)
        FileManager.default.createFile(atPath: url.path, contents: nil)
        let file = try FileHandle(forWritingTo: url)
        defer { try? file.close() }

        func le32(_ v: UInt32) -> Data { withUnsafeBytes(of: v.littleEndian) { Data($0) } }
        func le16(_ v: UInt16) -> Data { withUnsafeBytes(of: v.littleEndian) { Data($0) } }

        var header = Data()
        header.append("RIFF".data(using: .ascii)!)
        header.append(le32(0)) // placeholder: fileSize-8, 回填
        header.append("WAVE".data(using: .ascii)!)
        header.append("fmt ".data(using: .ascii)!)
        header.append(le32(16)) // fmt chunk size
        header.append(le16(1))  // PCM
        header.append(le16(UInt16(channels)))
        header.append(le32(UInt32(sampleRate)))
        header.append(le32(UInt32(byteRate)))
        header.append(le16(UInt16(blockAlign)))
        header.append(le16(UInt16(bitsPerSample)))
        header.append("data".data(using: .ascii)!)
        header.append(le32(0)) // placeholder: dataSize, 回填
        try file.write(contentsOf: header)

        // 逐块写 PCM 数据:直接从 CMBlockBuffer 取原始字节(已是 Linear PCM)
        var totalBytes = 0
        while let buf = readerOut.copyNextSampleBuffer() {
            if let dataBuf = CMSampleBufferGetDataBuffer(buf) {
                var length = 0
                var ptr: UnsafeMutablePointer<Int8>?
                CMBlockBufferGetDataPointer(dataBuf, atOffset: 0,
                                           lengthAtOffsetOut: nil,
                                           totalLengthOut: &length,
                                           dataPointerOut: &ptr)
                if let p = ptr, length > 0 {
                    let chunk = Data(bytes: p, count: length)
                    try file.write(contentsOf: chunk)
                    totalBytes += length
                }
            }
        }

        if reader.status == .failed {
            throw reader.error ?? MSError("PCM 读取失败")
        }

        // 回填 RIFF 大小和 dataSize
        try file.seek(toOffset: 4)
        try file.write(contentsOf: le32(UInt32(36 + totalBytes)))  // fileSize-8
        try file.seek(toOffset: 40)
        try file.write(contentsOf: le32(UInt32(totalBytes)))       // dataSize
    }
}

/// 识别回调 → continuation 的单次恢复闸门
private final class ResumeGate: @unchecked Sendable {
    private let lock = NSLock()
    private var resumed = false
    private var task: SFSpeechRecognitionTask?
    private var text = ""
    private let cont: CheckedContinuation<String, Error>

    init(_ cont: CheckedContinuation<String, Error>) { self.cont = cont }

    func attach(_ t: SFSpeechRecognitionTask) {
        lock.lock(); task = t; lock.unlock()
    }

    func absorb(_ s: String) { lock.lock(); text = s; lock.unlock() }

    func finish(error: Error?) {
        lock.lock()
        if resumed { lock.unlock(); return }
        resumed = true
        let value = text
        lock.unlock()
        if let error { cont.resume(throwing: error) } else { cont.resume(returning: value) }
    }
}
