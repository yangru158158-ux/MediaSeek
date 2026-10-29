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
            guard let y = Int(c[0]), let m = Int(c[1]), let d = Int(c[2]) else { return nil }
            return dayRange(y, m, d, "\(y)年\(m)月\(d)日")
        }) { return r }
        if let r = attempt("(\\d{4})年(\\d{1,2})月(份)?", { c in
            guard let y = Int(c[0]), let m = Int(c[1]) else { return nil }
            return monthRange(y, m, "\(y)年\(m)月")
        }) { return r }
        if let r = attempt("(\\d{1,2})月(\\d{1,2})[日号]", { c in
            // 未写年份:取当前年,若在未来则取去年
            guard let m = Int(c[0]), let d = Int(c[1]) else { return nil }
            if let r = dayRange(thisYear, m, d, "\(thisYear)年\(m)月\(d)日"), r.start <= now {
                return r
            }
            return dayRange(thisYear - 1, m, d, "\(thisYear - 1)年\(m)月\(d)日")
        }) { return r }
        if let r = attempt("(\\d{4})年", { c in Int(c[0]).flatMap { yearRange($0, "\($0)年") } }) { return r }
        if let r = attempt("(\\d{1,2})月(份)?", { c in
            guard let m = Int(c[0]) else { return nil }
            if let r = monthRange(thisYear, m, "\(thisYear)年\(m)月"), r.start <= now { return r }
            return monthRange(thisYear - 1, m, "\(thisYear - 1)年\(m)月")
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
        // 年份区间:「2015年到2026年」=跨年整段(2015-01-01 起,2027-01-01 止)
        if let r = attempt("(\\d{4})\\s*年?\\s*(?:到|至|~|～|—|–)\\s*(\\d{4})\\s*年?", { c in
            guard let y1 = Int(c[0]), let y2 = Int(c[1]),
                  y1 < y2, y1 >= 1950, y2 <= thisYear + 1,
                  let s = cal.date(from: DateComponents(year: y1, month: 1, day: 1)),
                  let e = cal.date(from: DateComponents(year: y2 + 1, month: 1, day: 1)) else { return nil }
            return QueryDateRange(start: s, end: e, label: "\(y1)年到\(y2)年")
        }) { return r }
        // 第七十四批:裸四位数年(「2026 杨如」/「语义:2026 杨如」)——用户惯性
        // 输年常不带「年」字;此前不解析,裸搜下 2026 退化为文字词(只出含
        // "2026"字样的截图),语义: 前缀下干脆无通道消费(装机实测 498 结果
        // 全是人名标签照片,与直接搜人名无异)。零宽边界防误吃:前后贴数字
        // 或日期分隔符(.-/~—–)及「至/到」都不匹配——「2026-10-26」
        // 「10/2026」「2026-2028」整词保留(验收 medium 实锤:裸数字断言
        // 防不了分隔符日期);整条查询只剩年份时不解析——保留「2026」作
        // 内容词的字面搜索,年份只与其他词共存时才升格为时间过滤。
        // 区间措辞:正则与 build 守卫同宽,覆盖 1900-2099
        if let r = attempt("(?<![0-9.\\-/~—–至到])((?:19|20)\\d{2})(?![0-9.\\-/~—–至到])", { c in
            guard let y = Int(c[0]), (1900...2099).contains(y) else { return nil }
            return yearRange(y, "\(y)年")
        }), !r.1.isEmpty {
            return r
        }
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
    private static let cache = NSCache<NSString, CLCircularRegion>()

    /// 地名 → 圆形区域(带缓存;离线/失败返回 nil,搜索照常降级)。
    /// 「在X」用默认 20km;裸城市名浏览用 50km(城市建成区+郊区一站式覆盖)。
    /// 批100:半径随地名自身尺度——placemark.region 反映行政区真实范围
    /// (城市 ~10-20km,国家/群岛数百 km)。此前硬编码 20/50km 圆:马尔代夫
    /// 是南北 ~870km 的 26 组环礁,50km 圆只盖住马累周边——251 装机实测
    /// 裸搜仅 33 个,各环礁度假村照片全被圆外切掉(西藏/呼伦贝尔/黄山
    /// 同病)。现取「真实范围与调用方下限」较大者,800km 封顶;placemark
    /// 范围缺失回退调用方半径——城市零变化(杭州 extent<50km 仍 50km)
    static func region(for name: String, radius: CLLocationDistance = 20_000) async -> CLCircularRegion? {
        let cacheKey = name as NSString   // 批100:半径由地名自身决定,缓存键去 radius
        if let v = cache.object(forKey: cacheKey) { return v }
        let region: CLCircularRegion? = await withCheckedContinuation { cont in
            let geocoder = CLGeocoder()
            geocoder.geocodeAddressString(name) { placemarks, error in
                if let pm = placemarks?.first, let loc = pm.location {
                    // placemark.region 是 CLRegion 基类,radius 在 CLCircularRegion
                    let extent = (pm.region as? CLCircularRegion)?.radius ?? radius
                    let r = min(max(extent, radius), 800_000)
                    // 批101:地理编码全记录——253 装机判别实锤 墨尔本✓/悉尼✗/
                    // 澳大利亚✗/澳洲(不在册):中文名可能解析失败或解析到
                    // 错误同名地(悉尼→加拿大新斯科舍?)。国家/行政区/坐标/
                    // 范围/最终半径全打 diag.log,装机重现一次即可定案
                    IndexingCoordinator.DiagLog.shared.event(
                        String(format: "地理编码[%@]→%@/%@ @%.2f,%.2f extent=%.0fk 最终=%.0fk",
                               name, pm.country ?? "?", pm.administrativeArea ?? "?",
                               loc.coordinate.latitude, loc.coordinate.longitude,
                               extent / 1000, r / 1000))
                    cont.resume(returning: CLCircularRegion(
                        center: loc.coordinate, radius: r, identifier: name))
                } else {
                    IndexingCoordinator.DiagLog.shared.event(
                        "地理编码[\(name)]✗ 无结果 err=\(error?.localizedDescription ?? "nil")")
                    cont.resume(returning: nil)
                }
            }
        }
        if let region { cache.setObject(region, forKey: cacheKey) }
        return region
    }
}
