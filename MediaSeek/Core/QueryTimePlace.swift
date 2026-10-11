import Foundation
import CoreLocation

/// 自然语言时间区间,如「2026年9月」「近7天」「去年」
struct QueryDateRange {
    let start: Date
    let end: Date
    let label: String
}

/// 自然语言时间解析:命中即从查询剥离,剩余词继续走内容通道;
/// 剩余为空且时间命中 → 纯时间浏览(按拍摄时间倒序列出该时段内容)。
enum TimeQueryParser {
    static func parse(_ text: String) -> (range: QueryDateRange?, remaining: String) {
        let cal = Calendar.current
        let now = Date()
        let thisYear = cal.component(.year, from: now)

        func dayRange(_ y: Int, _ m: Int, _ d: Int, _ label: String) -> QueryDateRange? {
            guard let s = cal.date(from: DateComponents(year: y, month: m, day: d)),
                  let e = cal.date(byAdding: .day, value: 1, to: s) else { return nil }
            return QueryDateRange(start: s, end: e, label: label)
        }
        func monthRange(_ y: Int, _ m: Int, _ label: String) -> QueryDateRange? {
            guard let s = cal.date(from: DateComponents(year: y, month: m, day: 1)),
                  let e = cal.date(byAdding: .month, value: 1, to: s) else { return nil }
            return QueryDateRange(start: s, end: e, label: label)
        }
        func yearRange(_ y: Int, _ label: String) -> QueryDateRange? {
            guard let s = cal.date(from: DateComponents(year: y, month: 1, day: 1)),
                  let e = cal.date(byAdding: .year, value: 1, to: s) else { return nil }
            return QueryDateRange(start: s, end: e, label: label)
        }
        func span(days n: Int, _ label: String) -> QueryDateRange? {
            let end = cal.startOfDay(for: now).addingTimeInterval(TimeInterval(86_400))
            let start = end.addingTimeInterval(-TimeInterval(n) * 86_400)
            return QueryDateRange(start: start, end: end, label: label)
        }
        func dayShift(_ offset: Int, _ label: String) -> QueryDateRange? {
            let d = cal.date(byAdding: .day, value: offset, to: now)!
            return dayRange(cal.component(.year, from: d),
                            cal.component(.month, from: d),
                            cal.component(.day, from: d), label)
        }

        // 按模式试匹配:命中即构造区间并从原文剥离该段
        func attempt(_ pattern: String, _ build: ([String]) -> QueryDateRange?) -> (QueryDateRange, String)? {
            guard let re = try? NSRegularExpression(pattern: pattern) else { return nil }
            let ns = text as NSString
            guard let m = re.firstMatch(in: text, range: NSRange(location: 0, length: ns.length)) else { return nil }
            let caps = (1..<m.numberOfRanges).map { i -> String in
                m.range(at: i).location == NSNotFound ? "" : ns.substring(with: m.range(at: i))
            }
            guard let r = build(caps) else { return nil }
            let remaining = ns.replacingCharacters(in: m.range, with: " ").trimmingCharacters(in: .whitespaces)
            return (r, remaining)
        }

        if let r = attempt("(近|最近|过去)(\\d{1,3})(天|日)", { c in Int(c[1]).flatMap { span(days: $0, "近\($0)天") } }) { return r }
        if let r = attempt("(近|最近|过去)(\\d{1,2})(周|星期|礼拜)", { c in Int(c[1]).flatMap { span(days: $0 * 7, "近\($0)周") } }) { return r }
        if let r = attempt("(近|最近|过去)(\\d{1,2})(个)?月", { c in
            Int(c[1]).flatMap { n in
                guard let s = cal.date(byAdding: .month, value: -n, to: now) else { return nil }
                return QueryDateRange(start: s, end: now, label: "近\(n)个月")
            }
        }) { return r }
        if let r = attempt("前天", { _ in dayShift(-2, "前天") }) { return r }
        if let r = attempt("昨天|昨日", { _ in dayShift(-1, "昨天") }) { return r }
        if let r = attempt("今天|今日", { _ in dayShift(0, "今天") }) { return r }
        if let r = attempt("(\\d{4})年(\\d{1,2})月(\\d{1,2})[日号]", { c in
            (Int(c[0]), Int(c[1]), Int(c[2])).flatMap { dayRange($0, $1, $2, "\(c[0])年\(c[1])月\(c[2])日") }
        }) { return r }
        if let r = attempt("(\\d{4})年(\\d{1,2})月(份)?", { c in
            (Int(c[0]), Int(c[1])).flatMap { monthRange($0, $1, "\(c[0])年\(c[1])月") }
        }) { return r }
        if let r = attempt("(\\d{1,2})月(\\d{1,2})[日号]", { c in
            // 未写年份:取当前年,若在未来则取去年
            (Int(c[0]), Int(c[1])).flatMap { m, d in
                if let thisYear = dayRange(thisYear, m, d, "\(thisYear)年\(m)月\(d)日"), thisYear.start <= now {
                    return thisYear
                }
                return dayRange(thisYear - 1, m, d, "\(thisYear - 1)年\(m)月\(d)日")
            }
        }) { return r }
        if let r = attempt("(\\d{4})年", { c in Int(c[0]).flatMap { yearRange($0, "\($0)年") } }) { return r }
        if let r = attempt("(\\d{1,2})月(份)?", { c in
            Int(c[0]).flatMap { m in
                if let r = monthRange(thisYear, m, "\(thisYear)年\(m)月"), r.start <= now { return r }
                return monthRange(thisYear - 1, m, "\(thisYear - 1)年\(m)月")
            }
        }) { return r }
        if let r = attempt("上(个)?(星期|礼拜|周)", { _ in
            guard let cur = cal.dateInterval(of: .weekOfYear, for: now) else { return nil }
            let start = cur.start.addingTimeInterval(-7 * 86_400)
            return QueryDateRange(start: start, end: cur.start, label: "上周")
        }) { return r }
        if let r = attempt("上(个)?月", { _ in
            let d = cal.date(byAdding: .month, value: -1, to: now)!
            return monthRange(cal.component(.year, from: d), cal.component(.month, from: d), "上个月")
        }) { return r }
        if let r = attempt("本月|这个月", { _ in monthRange(thisYear, cal.component(.month, from: now), "本月") }) { return r }
        if let r = attempt("今年|本年", { _ in yearRange(thisYear, "今年") }) { return r }
        if let r = attempt("去年", { _ in yearRange(thisYear - 1, "去年") }) { return r }
        if let r = attempt("前年", { _ in yearRange(thisYear - 2, "前年") }) { return r }
        return (nil, text)
    }
}

/// 自然语言地点:抽取「在XX(拍的…)」形态,CLGeocoder 地理编码成功才算命中
/// (失败不剥离,查询原样走内容通道,不误伤普通词)
enum PlaceQueryParser {
    static func extract(_ text: String) -> (name: String, remaining: String)? {
        guard let re = try? NSRegularExpression(pattern: "在([一-龥A-Za-z]{2,10})") else { return nil }
        let ns = text as NSString
        guard let m = re.firstMatch(in: text, range: NSRange(location: 0, length: ns.length)) else { return nil }
        var name = ns.substring(with: m.range(at: 1))
        while name.count > 2, name.hasSuffix("的") || name.hasSuffix("拍") {
            name = String(name.dropLast())
        }
        guard name.count >= 2 else { return nil }
        let remaining = ns.replacingCharacters(in: m.range, with: " ").trimmingCharacters(in: .whitespaces)
        return (name, remaining)
    }
}

enum PlaceGeocoder {
    private static let cache = NSCache<NSString, NSValue>()

    /// 地名 → 半径 20km 的圆形区域(带缓存;离线/失败返回 nil,搜索照常降级)
    static func region(for name: String) async -> CLCircularRegion? {
        if let v = cache.object(forKey: name as NSString) { return v as? CLCircularRegion }
        let region: CLCircularRegion? = await withCheckedContinuation { cont in
            let geocoder = CLGeocoder()
            geocoder.geocodeAddressString(name) { placemarks, _ in
                if let loc = placemarks?.first?.location {
                    cont.resume(returning: CLCircularRegion(
                        center: loc.coordinate, radius: 20_000, identifier: name))
                } else {
                    cont.resume(returning: nil)
                }
            }
        }
        if let region { cache.setObject(region as NSValue, forKey: name as NSString) }
        return region
    }
}
