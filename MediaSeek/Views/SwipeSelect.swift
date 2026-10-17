import SwiftUI

// 划选(第二十批起;第三十一批重设计):选择模式下**按住 ~0.3 秒**再划,
// 划过的条目批量勾选/取消;不按住的直接扫=滚动,手势完全不参与。
// 演进:二十批逐条目挂手势(ios26 堵死滚动)→二十三批撤到容器+方向门
// (横起手才算划选)——但手势一旦识别仍抢走滚动手,选择模式里竖向
// 滚不动(用户实测「选择项目时候内容需要往下滑,滑不动」)。方向判别
// 天生歧义,改意图判别:LongPress+Drag 序列手势,长按 0.3s 内移动
// ≤10pt 才武装——轻扫时序列手势直接失败,滚动 100% 畅通。
// 列表(List 全宽行,如资料库文件列表)与横向 ScrollView(查重组)不适用
// 划选——与滚动同向无法区分;划选只装在网格(搜索结果/人物组)。

/// 一次划选手势的进行态
struct SwipeSelectState {
    var active = false
    var selecting = true            // true=划选 / false=划反选
    var touched: Set<String> = []

    mutating func reset() {
        active = false
        touched = []
    }

    /// 武装(长按达成):此后划过的条目按"与起始格相反"的方向批量处理
    mutating func arm() {
        active = true
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
    /// 条目侧:选择模式时上报自身 frame(划选命中测试用)。非选择模式
    /// 连 frame 都不报,视图树零改动
    @ViewBuilder
    func swipeCellFrame(id: String, space: String, enabled: Bool) -> some View {
        if enabled {
            background(
                GeometryReader { g in
                    Color.clear.preference(key: SwipeCellFramesKey.self,
                                           value: [id: g.frame(in: .named(space))])
                }
            )
        } else {
            self
        }
    }

    /// 容器侧:整个网格挂一个"长按 0.3s + 拖动"序列手势。enabled=false 时
    /// 用手势掩码彻底关闭(.none)——不识别、不拦截,滚动不受任何影响。
    /// ⚠️ 选择模式里长按有别的用途的界面(结果卡 contextMenu/人物组诊断
    /// 长按),须在 enabled 时让位(调用方用条件守卫)
    func swipeSelectContainer(space: String, enabled: Bool,
                              onSweep: @escaping (DragGesture.Value) -> Void,
                              onEnd: @escaping () -> Void) -> some View {
        simultaneousGesture(
            LongPressGesture(minimumDuration: 0.3)
                .sequenced(before: DragGesture(minimumDistance: 4,
                                               coordinateSpace: .named(space)))
                .onChanged { v in
                    if case .second(true, let drag?) = v { onSweep(drag) }
                }
                .onEnded { _ in onEnd() },
            including: enabled ? .all : .none)
    }
}

/// 容器侧裁决:配合 @State var swipe/swipeFrames 使用
extension SwipeSelectState {
    mutating func handle(_ v: DragGesture.Value, frames: [String: CGRect],
                         isSelected: (String) -> Bool,
                         apply: (String, Bool) -> Void) {
        if !active {
            // 武装后的第一次移动:首触格定模式(与它的当前状态相反)
            arm()
            if let id = hit(v.location, frames: frames) {
                selecting = !isSelected(id)
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
