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

    /// AVAssetReader+Writer 直写 WAV PCM 16kHz 单声道。
    /// 第三十八批核心修:AVAssetExportSession+m4a 在真机上可能产出坏文件
    /// (StackOverflow 74454393),换 PCM 直写绕开——Speech 框架最可靠输入。
    private static func exportWAV(from avAsset: AVAsset, to url: URL, maxSeconds: Double) async throws {
        guard let audioTrack = try await avAsset.loadTracks(withMediaType: .audio).first else {
            throw MSError("无音轨")
        }
        // PCM 16kHz 单声道:Speech 框架推荐采样率
        let pcmSettings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: 16000.0,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsFloatKey: false,
            AVLinearPCMIsBigEndianKey: false,
        ]
        let reader = try AVAssetReader(asset: avAsset)
        let readerOut = AVAssetReaderTrackOutput(track: audioTrack, outputSettings: pcmSettings)
        reader.add(readerOut)

        // 限制时长:reader 的 timeRange
        reader.timeRange = CMTimeRange(
            start: .zero,
            duration: CMTime(seconds: maxSeconds, preferredTimescale: 600))

        let writer = try AVAssetWriter(outputURL: url, fileType: .wav)
        let writerIn = AVAssetWriterInput(mediaType: .audio, outputSettings: pcmSettings)
        writer.add(writerIn)

        guard reader.startReading() else {
            throw reader.error ?? MSError("音轨读取启动失败")
        }
        writer.startWriting()
        writer.startSession(atSourceTime: .zero)

        // 逐块搬运, writerInput 需要在自己的队列上喂
        let queue = DispatchQueue(label: "speech-wav-export")
        return try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            var finished = false
            writerIn.requestMediaDataWhenReady(on: queue) {
                guard !finished else { return }
                while writerIn.isReadyForMoreMediaData {
                    if let buf = readerOut.copyNextSampleBuffer() {
                        writerIn.append(buf)
                    } else {
                        writerIn.markAsFinished()
                        finished = true
                        writer.finishWriting { _ in
                            if writer.status == .completed {
                                cont.resume()
                            } else {
                                cont.resume(throwing: writer.error ?? MSError("WAV 写入失败"))
                            }
                        }
                        return
                    }
                }
            }
        }
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
