import CoreML
import Foundation

/// 批157(用户:「每次重启 app 都要重新加载各个模型……每次都要几分钟」):启动耗时账本 +
/// 计算单元开关。此前模型加载耗时没有任何记录——不知道「几分钟」卡在打开模型、分词器,
/// 还是第一次推理(神经引擎首跑要把编译好的程序装进引擎)。每个模型每一步的毫秒数与
/// 「距进程启动多久」写进 diag.log,装机冷启动一次就能定位。
///
/// **这个文件必须能单独编译**:FaceKit.swift / EmbeddingModels.swift 会被人脸测试等
/// 流水线单独拿去 swiftc,所以这里不直接引用 DiagLog(它住在 IndexingCoordinator.swift 里)
/// ——日志出口由 ModelManager 启动时装上,没装时静默丢弃。
enum ModelLoadTiming {
    /// 进程启动的近似时刻:第一次用到时记(App 启动后第一次加载模型就是这里)
    static let processStart = Date()
    /// 可变状态都装在带锁的类里(不用可变静态变量,同 IndexingCoordinator 的
    /// FirstErrorBox / Counter 写法)
    private static let firstInferenceLedger = NameLedger()
    private static let sinkBox = SinkBox()

    /// 日志出口(ModelManager.init 里装上:写 diag.log)
    static func setSink(_ sink: @escaping (String) -> Void) {
        sinkBox.set(sink)
    }

    /// 记一步耗时:diag.log 里一行 `模型加载[名字] 步骤 Nms(距进程启动 X.Xs)`
    static func log(_ model: String, _ step: String, from start: Date) {
        let ms = Int(Date().timeIntervalSince(start) * 1000)
        let line = String(format: "模型加载[%@] %@ %dms(距进程启动 %.1fs)",
                          model, step, ms, Date().timeIntervalSince(processStart))
        sinkBox.emit(line)
    }

    /// 第一次推理的耗时:每个模型只记一次
    static func firstInference(_ model: String, from start: Date) {
        if firstInferenceLedger.insertNew(model) { log(model, "首次推理", from: start) }
    }

    /// 模型用哪种计算单元:默认神经引擎;设置页「模型推理芯片」可切 GPU / 仅 CPU——
    /// **只做诊断用**:切了数值有细微差别,门槛与查询向量都是按神经引擎标定的,测完切回。
    /// 改了要重启 App 才生效(模型只在加载时读一次)
    static func configuration() -> MLModelConfiguration {
        let config = MLModelConfiguration()
        switch UserDefaults.standard.string(forKey: "modelComputeUnits") ?? "ane" {
        case "gpu": config.computeUnits = .cpuAndGPU
        case "cpu": config.computeUnits = .cpuOnly
        default: config.computeUnits = .cpuAndNeuralEngine
        }
        return config
    }
}

/// 带锁的名字集合:insertNew 返回「之前没有过」
private final class NameLedger: @unchecked Sendable {
    private let lock = NSLock()
    private var names = Set<String>()

    func insertNew(_ name: String) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        if names.contains(name) { return false }
        names.insert(name)
        return true
    }
}

/// 带锁的日志出口:没装出口时 emit 什么都不做
private final class SinkBox: @unchecked Sendable {
    private let lock = NSLock()
    private var sink: ((String) -> Void)?

    func set(_ s: @escaping (String) -> Void) {
        lock.lock()
        sink = s
        lock.unlock()
    }

    func emit(_ line: String) {
        lock.lock()
        let s = sink
        lock.unlock()
        s?(line)
    }
}
