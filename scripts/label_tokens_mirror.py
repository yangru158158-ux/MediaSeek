#!/usr/bin/env python3
"""labelTokens 全保真 Python 镜像(第四十八批词典直译免疫规则)。

此前两轮本机镜像与 CI 分歧的教训:english(for:) 查两张表(dictionary +
LabelZh.englishFromZh)、多命中按位置排序用空格拼接——镜像漏一环,CI 就红一轮。
本镜像从 Swift 源码直接解析三张表,逐语义复刻,跑 34 条断言中的标签相关
子集 + 两条 1566 词穷尽循环(CI 上从未跑完过,先在本机排雷)。
"""
import re
import sys

ROOT = "MediaSeek/Core/"
QU = open(ROOT + "QueryUnderstanding.swift", encoding="utf-8").read()
LZ = open(ROOT + "LabelZh.swift", encoding="utf-8").read()
LZE = open(ROOT + "LabelZhExtra.swift", encoding="utf-8").read()

# ---- 解析三张表 ----
dic_block = QU.split("static let dictionary: [(String, String)] = [", 1)[1]
dic_block = dic_block.split("\n    ]", 1)[0]
DICTIONARY = re.findall(r'\("([^"]+)",\s*"([^"]+)"\)', dic_block)

gen_block = QU.split("static let genericTokens: Set<String> = [", 1)[1]
gen_block = gen_block.split("]", 1)[0]
GENERIC = set(re.findall(r'"([a-z_]+)"', gen_block))

syn_block = QU.split("static let synonyms: [String: [String]] = [", 1)[1]
syn_block = syn_block.split("\n    ]", 1)[0]
SYNONYMS = {}
for k, vals in re.findall(r'"([^"]+)":\s*\[([^\]]*)\]', syn_block):
    SYNONYMS[k] = [v.strip().strip('"') for v in vals.split(",") if v.strip()]

base_block = LZ.split("private static let base: [String: String] = [", 1)[1]
base_block = base_block.split("\n    ]", 1)[0]
BASE = dict(re.findall(r'"([^"]+)":\s*"([^"]+)"', base_block))
extra_block = LZE.split("= [", 1)[1]
extra_block = extra_block.split("\n    ]", 1)[0]
EXTRA = dict(re.findall(r'"([^"]+)":\s*"([^"]+)"', extra_block))
ZH = {**EXTRA, **BASE}          # base 优先(merge { a, _ in a })
EN_FROM_ZH = {}
for en, cn in list(EXTRA.items()) + list(BASE.items()):  # base 后写先到先得
    pass
for en, cn in list(BASE.items()) + list(EXTRA.items()):
    EN_FROM_ZH.setdefault(cn, en)


def term_coverage(term):
    """第六十二批与 Swift termCoverage 同构:碎片匹配→长度降序贪心选区间
    (只剔严格内含,同区间多译词都留)→连接字容忍的覆盖判定。
    返回 (kind, longest_run, spans):kind=no(零命中)/partial(盖不满,
    longest_run=最长连续生字符段,连接字与已覆盖字符断开计数)/complete。"""
    CONNECTIVES = set("的了与和及或在是")
    matches = []  # (start, end, en) 字符偏移
    tables = list(DICTIONARY) + list(EN_FROM_ZH.items())
    for zh, en in tables:
        if len(zh) == 1:
            if term == zh:
                matches.append((0, len(term), en))
            continue
        pos = term.find(zh)
        if pos >= 0:
            matches.append((pos, pos + len(zh), en))
    if not matches:
        return ("no", 0, [])
    matches.sort(key=lambda m: (-(m[1] - m[0]), m[0]))
    spans = []
    for m in matches:
        if any(s[0] <= m[0] and m[1] <= s[1] and (s[1] - s[0]) > (m[1] - m[0])
               for s in spans):
            continue
        spans.append(m)
    covered = set()
    for s in spans:
        covered.update(range(s[0], s[1]))
    longest_run = run = 0
    for i, c in enumerate(term):
        if i not in covered and c not in CONNECTIVES:
            run += 1
            longest_run = max(longest_run, run)
        else:
            run = 0
    if longest_run > 0:
        return ("partial", longest_run, spans)
    return ("complete", 0, sorted(spans, key=lambda m: (m[0], m[1])))


def english_for(core):
    """第六十批覆盖率规则逐语义复刻(经 term_coverage):盖满才捐,同词
    全 core 全局只捐一次(验收发现曾漏更新,靠捐超集假绿)。"""
    out = []
    seen = set()
    for term in core.split():
        if not term:
            continue
        kind, _, spans = term_coverage(term)
        if kind != "complete":
            continue   # 盖不满/零命中:不捐
        for m in spans:
            if m[2] not in seen:
                seen.add(m[2])
                out.append(m[2])
    return " ".join(out) if out else None


def is_proper_noun(term):
    """第六十二批:连续生字符≥2=专名(华研),整词只走文字通道;
    单生字符(红裙子的红)与零命中(财报)不判,保持原文兜底。"""
    kind, longest_run, _ = term_coverage(term)
    return kind == "partial" and longest_run >= 2


def is_dictionary_unknown(term):
    """第六十四批人名闸判据:真零命中(无任何碎片)。english_for()==None
    比它宽(还涵盖盖不满的修饰词红裙子),闸只认这个窄口径。"""
    kind, _, _ = term_coverage(term)
    return kind == "no"


def expanded(tokens):
    out = set(tokens)
    for t in tokens:
        out |= set(SYNONYMS.get(t, []))
    return out


# 第七十二批:繁→简字表解析(定义必须先于 label_tokens 的首次调用)
t2s_block = QU.split("static let t2sChars: [Character: Character] = [", 1)[1]
t2s_block = t2s_block.split("\n    ]", 1)[0]
T2S = dict(re.findall(r'"([^"]+)":\s*"([^"]+)"', t2s_block))
def to_simplified(s):
    return "".join(T2S.get(ch, ch) for ch in s)


def label_tokens(query):
    # 第七十二批二修(验收 low):与 Swift labelTokens 入口同规繁→简归一——
    # 镜像对繁体查询恢复保真,Swift 侧归一行为漂移时此处能拦
    query = to_simplified(query)
    raw = {w for w in re.split(r"\s+", query.lower()) if w}
    tokens = set(raw)
    dict_tokens = set()
    en = english_for(query)
    if en:
        for t in en.lower().split(" "):
            if t:
                tokens.add(t)
                dict_tokens.add(t)
    if len(query) >= 3:
        typed = {t for t in raw if all(ord(c) < 128 for c in t)}
        typed_all_generic = bool(typed) and typed.issubset(GENERIC)
        if typed_all_generic:
            tokens = expanded(tokens)
        else:
            tokens = expanded(tokens) - (GENERIC - dict_tokens)
    else:
        # 第五十六批:短词短路只护种子词,扩展注入的泛化词照剔
        seed = set(tokens)
        tokens = expanded(tokens) - (GENERIC - seed)
    return tokens


def expect(cond, msg):
    if not cond:
        print("FAIL:", msg)
        sys.exit(1)


# ---- 34 条断言中的标签相关子集 ----
expect("machine" in label_tokens("machine"), "machine 本尊保留")
expect("person" in label_tokens("person"), "person 本尊保留")
expect("machine" in label_tokens("机器"), "机器(2字)直译保留")
expect("screenshot" in label_tokens("屏幕截图"), "屏幕截图→screenshot 免疫(5197张)")
expect("machine" in label_tokens("机器设备"), "机器设备混合直译 machine 免疫(1306张)")
expect("screenshot" in label_tokens("网页"), "网页→website screenshot 保留")
expect("people" in label_tokens("朋友"), "朋友→friend people 保留")
expect("document" in label_tokens("pdf"), "pdf→document 保留")
expect("beach" in label_tokens("machine beach"), "machine beach 留 beach")
expect("machine" not in label_tokens("machine beach"), "machine beach 剔 machine(手敲防灌水)")
expect("people" not in label_tokens("people beach photo"), "people beach photo 剔 people")
expect("boy" in expanded({"man"}) and "guy" in expanded({"man"}), "man 扩展含 boy/guy")
expect("lady" in expanded({"woman"}), "woman 扩展含 lady")
expect("adult" not in expanded({"woman"}), "woman 扩展不含 adult")
expect("gentleman" in expanded({"man"}), "man 扩展含 gentleman")

# ---- 穷尽循环排雷(CI 从未跑完) ----
multiword = [en for en in ZH if " " in en]
print(f"多词英文键(含空格,穷尽循环 contains(en) 会误杀): {multiword or '无'}")
zh_to_ens = {}
for en, cn in ZH.items():
    zh_to_ens.setdefault(cn, []).append(en)

fail_en = [en for en in ZH if en not in label_tokens(en)]
expect(not fail_en, f"英文标签直搜被剔除: {fail_en[:5]}")
print(f"英文直搜穷尽: {len(ZH) - len(fail_en)}/{len(ZH)} 全保留")

fail_cn = []
for cn, ens in zh_to_ens.items():
    toks = label_tokens(cn)
    if not any(e in toks for e in ens):
        fail_cn.append((cn, ens))
expect(not fail_cn, f"中文反查哑火: {fail_cn[:5]}")
print(f"中文反查穷尽: {len(zh_to_ens) - len(fail_cn)}/{len(zh_to_ens)} 全命中")

# ---- 第六十二批:专名判定对拍(与 Swift 断言同集) ----
expect(is_proper_noun("华研机器人"), "华研机器人:华研 2 连续生字符 → 专名")
expect(not is_proper_noun("机器设备"), "机器设备全覆盖=复合词,非专名")
expect(not is_proper_noun("妈妈的生日"), "连接字不算生字符,非专名")
expect(not is_proper_noun("屏幕和截图"), "「和」连接字,非专名")
expect(not is_proper_noun("机器人"), "词典条目自身全覆盖,非专名")
expect(not is_proper_noun("woman"), "ASCII 零命中非专名")
expect(not is_proper_noun("财报"), "中文零命中非专名(词典没收)")
expect(english_for("红裙子") is None, "红裙子盖不满不捐(60 批规则不变)")
expect(not is_proper_noun("红裙子"), "单生字符「红」=修饰词,不判专名")
# ---- 第六十四批:人名闸零命中判据对拍(与 Swift 断言同集) ----
expect(is_dictionary_unknown("管雯"), "管雯=真零命中(人名)")
expect(is_dictionary_unknown("财报"), "财报=零命中(普通词)")
expect(is_dictionary_unknown("woman"), "裸英文=零命中")
expect(not is_dictionary_unknown("红裙子"), "红裙子=partial,不是零命中")
expect(not is_dictionary_unknown("机器人"), "机器人=complete,不是零命中")
expect(not is_dictionary_unknown("华研机器人"), "华研机器人=专名,不是零命中")
fail_pn = [zh for zh, _ in DICTIONARY if is_proper_noun(zh)] \
    + [zh for zh in EN_FROM_ZH if is_proper_noun(zh)]
expect(not fail_pn, f"词条自身被判专名(覆盖评估回归): {fail_pn[:5]}")
print(f"专名不变量: 词典 {len(DICTIONARY)} 条 + 反查 {len(EN_FROM_ZH)} 词 全部非专名")

# ---- 第六十九批:英文证件桥对拍(与 Swift 断言同集,表从源码解析) ----
cc_block = QU.split("static let chineseCounterparts: [String: String] = [", 1)[1]
COUNTERPARTS = dict(re.findall(r'"([^"]+)":\s*"([^"]+)"', cc_block.split("\n    ]", 1)[0]))
al_block = QU.split("static let ocrAliases: [String: [String]] = [", 1)[1]
al_block = al_block.split("\n    ]", 1)[0]
OCR_ALIASES = {k: re.findall(r'"([^"]+)"', vals)
               for k, vals in re.findall(r'"([^"]+)":\s*\[([^\]]*)\]', al_block)}
expect(COUNTERPARTS.get("id") == "身份证" and COUNTERPARTS.get("ID") is None,
       "镜像口径:表键全小写,id→身份证(大小写归一在函数层)")
expect(COUNTERPARTS.get("identity card") == "身份证"
    and COUNTERPARTS.get("identity document") == "身份证",
       "identity 族短语键在表")
expect(COUNTERPARTS.get("passport") == "护照", "passport→护照")
expect(COUNTERPARTS.get("licence") == "驾照" and COUNTERPARTS.get("license") == "驾照",
       "licence/license→驾照(71 批翻案+74 批美式对称面,用户批准)")
expect("driving" not in COUNTERPARTS and "card" not in COUNTERPARTS,
       "driving/card 泛词不入表")
def bridged_variants(term):
    zh = COUNTERPARTS.get(term.lower().strip())
    return [zh] + OCR_ALIASES.get(zh, []) if zh else []
expect(bridged_variants("identity") == ["身份证", "公民身份", "居民身份证", "签发机关", "身份證"],
       "bridgedVariants(identity)=身份证+全部别名(含繁体)")
expect(bridged_variants("licence") == ["驾照", "准驾车型", "机动车驾驶证"],
       "licence 桥接变体=驾照+别名(第七十一批)")
expect(bridged_variants("beach") == [], "表外词零追加")
expect(OCR_ALIASES.get("驾照") == OCR_ALIASES.get("驾驶证"),
       "驾照/驾驶证别名同表(兄弟缺口已补)")
expect(dict(DICTIONARY).get("护照") == "passport", "护照词典对已补(语义模板)")

# ---- 第七十二批:繁→简归一对拍(表定义已前置,此处结构不变量守恒) ----
expect(len(T2S) > 150 and all(k != v for k, v in T2S.items()),
       f"字表规模 {len(T2S)} 条,无键值同对的空转换")
expect(not (set(T2S.keys()) & set(T2S.values())),
       "无键同时是另一键的值(单遍无链式,幂等由构造保证)")
expect(to_simplified("駕駛證") == "驾驶证" and to_simplified("證件") == "证件"
    and to_simplified("護照") == "护照" and to_simplified("美團") == "美团",
       "繁→简:证件家族+股票名全链路走通")
expect(to_simplified("动物") == "动物" and to_simplified("美團 存款") == "美团 存款",
       "简体/同形透传,空格保留")
expect(OCR_ALIASES.get("证件") == ["身份证", "护照", "驾驶证"],
       "证件入别名表(与 licence 同权,221 装机实测 证件 语义0)")
expect(dict(DICTIONARY).get("驾驶证") == "driver license document",
       "驾驶证词典对补齐(与驾照同款模板)")
print(f"证件桥: 表 {len(COUNTERPARTS)} 键 / 别名 {len(OCR_ALIASES)} 组 对拍一致")

print("PASS: 镜像全绿(词典", len(DICTIONARY), "条/标签", len(ZH), "词/泛化", len(GENERIC), "个)")
