import SwiftUI
import UIKit

// 划选(第二十批起;第三十二批 UIKit 重写):选择模式下**按住 ~0.3 秒**再划,
// 划过的条目批量勾选/取消;不按住的直接扫=滚动,滚动永不失效。
// 演进史(每一步都是 iOS 26 实测教训):
//   二十批:逐条目挂手势 → 滚动彻底堵死
//   二十三批:撤到容器一个+方向门 → 选择模式里滚动仍被抢(SwiftUI 手势
//            一旦参与识别就抢 UIKit 滚动,与方向判断无关)
//   三十一批:LongPress+Drag 序列手势(意图判别)→ 划选可用,但滚动
//            还是死(SwiftUI 手势层与滚动不共存,iOS 26 无解)
//   三十二批:**改 UIKit UILongPressGestureRecognizer**——通过
//     UIGestureRecognizerDelegate.shouldRecognizeSimultaneouslyWith 与
//     UIScrollView 的滚动手势真正共存(相册类 App 的底层做法):
//     · 识别器挂在祖先 UIScrollView 上,enabled=false 时彻底禁用零干扰
//     · 按住 0.3s(允许漂移 30pt)武装,划动批量勾选,横竖皆可
//     · 直接扫=长按失败+滚动照常(UIKit 失败识别器让路,行为确定)
//     · 坐标:识别器报桥接视图(=网格)本地坐标,与 SwiftUI 命名空间
//       同为左上原点,直接命中测试 swipeFrames

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

    /// 容器侧:背景里放一座 UIKit 桥——识别器挂到祖先 UIScrollView 上,
    /// 与滚动共存;enabled=false 时识别器禁用,零干扰。
    /// ⚠️ 选择模式里长按有别的用途的界面(结果卡 contextMenu/人物组诊断
    /// 长按),须在 enabled 时让位(调用方条件守卫)
    func swipeSelectContainer(space: String, enabled: Bool,
                              onSweep: @escaping (CGPoint) -> Void,
                              onEnd: @escaping () -> Void) -> some View {
        background(
            SwipeSelectBridge(enabled: enabled, onPoint: onSweep, onEnd: onEnd)
                .allowsHitTesting(false)
        )
    }
}

/// UIKit 手势桥:纯挂载点视图(不参与命中测试),把 UILongPressGestureRecognizer
/// 挂到层级里的祖先 UIScrollView 上
struct SwipeSelectBridge: UIViewRepresentable {
    let enabled: Bool
    let onPoint: (CGPoint) -> Void
    let onEnd: () -> Void

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    func makeUIView(context: Context) -> UIView {
        let v = UIView()
        let rec = UILongPressGestureRecognizer(target: context.coordinator,
                                               action: #selector(Coordinator.fire(_:)))
        rec.minimumPressDuration = 0.3
        rec.allowableMovement = 30
        rec.delegate = context.coordinator
        context.coordinator.recognizer = rec
        DispatchQueue.main.async { [weak v, weak rec] in
            // 此刻桥视图已在层级里:向上找到祖先 UIScrollView 再挂——
            // 挂在祖先上才能看到落在内容上的触摸;锚点=桥视图(=网格),
            // 识别器回报的 scrollView 坐标要换算回网格本地
            var sv = v?.superview
            while let s = sv, !(s is UIScrollView) { sv = s.superview }
            if let scroll = sv as? UIScrollView, let rec, let v {
                scroll.addGestureRecognizer(rec)
                context.coordinator.anchor = v
                rec.isEnabled = self.enabled
            }
        }
        return v
    }

    func updateUIView(_ uiView: UIView, context: Context) {
        context.coordinator.parent = self
        // enabled 翻转只动开关;识别器从 ScrollView 摘除交给 SwiftUI 生命周期
        context.coordinator.recognizer?.isEnabled = enabled
    }

    final class Coordinator: NSObject, UIGestureRecognizerDelegate {
        var parent: SwipeSelectBridge
        var recognizer: UILongPressGestureRecognizer?
        weak var anchor: UIView?   // 桥视图=网格本体,坐标换算锚点

        init(_ p: SwipeSelectBridge) { parent = p }

        @objc func fire(_ r: UILongPressGestureRecognizer) {
            // 换算到网格本地坐标(=SwiftUI 命名空间,左上原点)
            guard let v = anchor else { return }
            switch r.state {
            case .began, .changed:
                parent.onPoint(r.location(in: v))
            default:
                parent.onEnd()
            }
        }

        /// 与一切手势(尤其滚动)共存——这是 SwiftUI 手势层给不了的
        func gestureRecognizer(_ g: UIGestureRecognizer,
                               shouldRecognizeSimultaneouslyWith other: UIGestureRecognizer) -> Bool {
            true
        }
    }
}

/// 容器侧裁决:配合 @State var swipe/swipeFrames 使用
extension SwipeSelectState {
    mutating func handle(_ p: CGPoint, frames: [String: CGRect],
                         isSelected: (String) -> Bool,
                         apply: (String, Bool) -> Void) {
        if !active {
            // 武装后的第一个点:首触格定模式(与它的当前状态相反)
            arm()
            if let id = hit(p, frames: frames) {
                selecting = !isSelected(id)
            }
        }
        guard let id = hit(p, frames: frames),
              !touched.contains(id) else { return }
        touched.insert(id)
        apply(id, selecting)
    }

    private func hit(_ p: CGPoint, frames: [String: CGRect]) -> String? {
        frames.first { $0.value.insetBy(dx: -3, dy: -3).contains(p) }?.key
    }
}
