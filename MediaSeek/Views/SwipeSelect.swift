import SwiftUI

// 划选(第二十批,用户点名「需要做选择的地方可以划过图片/文件/视频做选择」):
// 相册手感——选择模式下按住横向划过一排条目即批量勾选/取消。
// ① 首个触点定模式:按在未选项上=划选,按在已选项上=划反选
// ② 横向起手才启动(位移 dx ≥ dy×1.3);竖向移动留给 ScrollView 滚动,
//    两者互不抢事件(simultaneousGesture)
// ③ 普通点按不受影响:drag 需移动 12pt 才算划,单击仍是原 toggle
// 列表(List 全宽行,如资料库文件列表)与横向 ScrollView(查重组)不适用
// 划选——竖划/横划分别与滚动彻底同向,无法区分,苹果文件 App 同样只做
// 点选+全选;划选只装在网格(搜索结果/人物组)。

/// 一次划选手势的进行态
struct SwipeSelectState {
    var active = false
    var selecting = true            // true=划选 / false=划反选
    var touched: Set<String> = []

    mutating func reset() {
        active = false
        touched = []
    }
}

/// 条目 frame 上报(LazyVGrid 只给屏内条目装 frame——划选也只发生在屏内)
struct SwipeCellFramesKey: PreferenceKey {
    static var defaultValue: [String: CGRect] = [:]
    static func reduce(value: inout [String: CGRect], nextValue: () -> [String: CGRect]) {
        value.merge(nextValue()) { _, new in new }
    }
}

extension View {
    /// 条目侧:上报自身 frame + 挂划选手势。enabled=false 时手势静默。
    /// onSweep 把 DragGesture.Value 交给容器统一裁决(方向门/命中测试/
    /// 模式判定都在容器层,条目只负责报 frame 和转发)
    func swipeSelectable(id: String, space: String, enabled: Bool,
                         onSweep: @escaping (DragGesture.Value) -> Void,
                         onEnd: @escaping () -> Void = {}) -> some View {
        background(
            GeometryReader { g in
                Color.clear.preference(key: SwipeCellFramesKey.self,
                                       value: [id: g.frame(in: .named(space))])
            }
        )
        .simultaneousGesture(
            DragGesture(minimumDistance: 12, coordinateSpace: .named(space))
                .onChanged { v in
                    guard enabled else { return }
                    onSweep(v)
                }
                .onEnded { _ in onEnd() }
        )
    }
}

/// 容器侧裁决:配合 @State var swipe/swipeFrames 使用
extension SwipeSelectState {
    /// 划选事件入口:横向起手才启动;命中过的条目不重复处理
    mutating func handle(_ v: DragGesture.Value, frames: [String: CGRect],
                         isSelected: (String) -> Bool,
                         apply: (String, Bool) -> Void) {
        if !active {
            // 竖向起手=用户在滚动,本次手势不再启动划选
            guard abs(v.translation.width) >= abs(v.translation.height) * 1.3 else { return }
            active = true
            touched = []
            if let id = hit(v.location, frames: frames) {
                selecting = !isSelected(id)   // 首触点定模式
            }
        }
        guard let id = hit(v.location, frames: frames),
              !touched.contains(id) else { return }
        touched.insert(id)
        apply(id, selecting)
    }

    private func hit(_ p: CGPoint, frames: [String: CGRect]) -> String? {
        frames.first { $0.value.insetBy(dx: -3, dy: -3).contains(p) }?.key
    }
}
