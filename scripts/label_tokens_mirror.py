#!/usr/bin/env python3
"""labelTokens 全保真 Python 镜像(第四十八批词典直译免疫规则)。

此前两轮本机镜像与 CI 分歧的教训:english(for:) 查两张表(dictionary +
LabelZh.englishFromZh)、多命中按位置排序用空格拼接——镜像漏一环,CI 就红一轮。
本镜像从 Swift 源码直接解析三张表,逐语义复刻,跑 Swift 断言里的标签相关
子集 + 标签词典穷尽(词集级与匹配级,各两个方向)+ 专名/证件桥/繁简对拍。
(批142 更正:此处原写「跑 34 条断言中的…」——34 是创建时点 Swift 侧的
断言总数,此后两边都涨过几轮,那个数早已不是任何一边的口径;批134 验收
low 点名,随本批改动一并更正为不带时点数字的表述。)
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
# 批142:反查全集(同中文名的全部键,键排序)——Swift LabelZh.englishKeysFromZh
# 同构。Swift 侧单值表 englishFromZh 重名先到先得、先到哪个随字典遍历顺序
# 变;镜像的 EN_FROM_ZH 是确定的(base 先),所以镜像复现不了那种漂移,
# 只能守「补全后全部键都在词集」
EN_KEYS_FROM_ZH = {}
for en, cn in list(BASE.items()) + list(EXTRA.items()):
    EN_KEYS_FROM_ZH.setdefault(cn, []).append(en)
EN_KEYS_FROM_ZH = {cn: sorted(keys) for cn, keys in EN_KEYS_FROM_ZH.items()}


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


def label_term_matches(query_tokens, term):
    """牌里单个词的命中判定(Swift labelTermMatches 同构,规则只此一处):
    简单词在词集才命中;复合词整键在词集(批142)或全部片段都在词集(批126)。"""
    if "_" in term:
        if term in query_tokens:
            return True
        return set(term.split("_")) <= query_tokens
    return term in query_tokens


def label_tokens_match_count(query_tokens, label_title):
    """批142:QueryUnderstanding.labelTokensMatchCount 逐语义复刻(标签牌匹配)。
    牌里的词按「, 」拆开逐个判;返回匹配词数。"""
    count = 0
    for term in re.split(r"[, ]", label_title.lower()):
        if term and label_term_matches(query_tokens, term):
            count += 1
    return count


def label_tokens_matched_terms(query_tokens, label_title):
    """批143仪器:QueryUnderstanding.labelTokensMatchedTerms 同构——返回牌里
    实际命中的词(与计数函数共用判定,个数恒等于它)。"""
    return [term for term in re.split(r"[, ]", label_title.lower())
            if term and label_term_matches(query_tokens, term)]


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
    # 批142:同名多键补全——Swift labelTokens 沿 english(for:) 实际捐出的碎片
    # 区间,把区间对应中文名下的全部键补进词集并按直译词对待(同享泛化剔除
    # 免疫);镜像同步。位置与 Swift 同:词典英译之后、下划线拆分之前。
    # 判据沿用 term_coverage(二修,验收 medium):盖满才捐、嵌套只留最长、
    # 单字键整词相等才生效——首版只认整词原子,短语里的多键名仍会漂移
    for t in raw:
        kind, _, spans = term_coverage(t)
        if kind != "complete":
            continue
        for m in spans:
            for k in EN_KEYS_FROM_ZH.get(t[m[0]:m[1]], []):
                tokens.add(k)
                dict_tokens.add(k)
    # 批131(批130验收 medium 同步):查询端下划线复合词拆分
    # —Swift 批130 加了(play_card→{play_card,play,card}),镜像同步
    # 批143(用户定「两个都精确」):本身就是词表里的键不拆片段——整键已能命中
    # 自己的牌,拆出的片段只会顺带命中同名的其它牌;词表外的下划线串照旧拆
    for t in raw:
        if "_" in t and t not in ZH:
            for frag in t.split("_"):
                if frag:
                    tokens.add(frag)
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


# ---- 标签相关子集(与 Swift 断言同集) ----
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
    # 批128收缩(用户确认):只桥中文对应词,不再追加 ocrAliases——
    # 驾照字段出现在所有提及驾照的文档,不只在驾照本体
    zh = COUNTERPARTS.get(term.lower().strip())
    return [zh] if zh else []
expect(bridged_variants("identity") == ["身份证"],
       "bridgedVariants(identity)=[身份证](批128收缩:只桥对应词)")
expect(bridged_variants("licence") == ["驾照"],
       "licence 桥接变体=驾照(批128收缩:去准驾车型/机动车驾驶证)")
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
expect(dict(DICTIONARY).get("驾驶证") == "driver license card",
       "驾驶证词典对补齐(批83 已 document→card,镜像滞后三批实锤)")
print(f"证件桥: 表 {len(COUNTERPARTS)} 键 / 别名 {len(OCR_ALIASES)} 组 对拍一致")

# ---- 批143:查询端下划线处理护栏(与 Swift 断言同集;批134 那条改写) ----
# 批130 加了拆分逻辑但两侧测试都没断言它;批143(用户定「两个都精确」):
# 词表内的键打进来自身要能命中自己的牌(整键),但不再拆片段——拆出的片段会
# 顺带命中同名的其它牌(3 (2) 装机:balloon_hotair 标签=56 对 热气球 标签=7)
pc = label_tokens("play_card")
expect("play_card" in pc and "play" not in pc and "card" not in pc,
       "play_card(词表内的键)整键进词集、不拆片段")
expect(label_tokens_match_count(pc, "play_card") == 1 and label_tokens_match_count(pc, "card") == 0,
       "play_card 命中自己的牌,不命中 card 牌")
idc = label_tokens("id_card")
expect("id" in idc and "card" in idc, "词表外的下划线串照旧拆片段(id_card→id+card)")
# 全量复合键:打进来都命中自己的牌;片段不由拆分注入(同义表引入的除外)
own_miss = [en for en in ZH if "_" in en and label_tokens_match_count(label_tokens(en), en) == 0]
expect(not own_miss, f"复合键打进来都命中自己的牌(整键): {own_miss[:3]}")
split_leak = []
for en in ZH:
    if "_" not in en:
        continue
    toks = label_tokens(en)
    via_syn = expanded({en})
    for frag in en.split("_"):
        if frag and frag in toks and frag not in via_syn:
            split_leak.append(f"{en}->{frag}")
expect(not split_leak, f"复合键的片段不由拆分注入(同义表引入的除外): {split_leak[:3]}")
term_mismatch = []
for en, cn in ZH.items():
    for toks in (label_tokens(en), label_tokens(cn)):
        if len(label_tokens_matched_terms(toks, en)) != label_tokens_match_count(toks, en):
            term_mismatch.append(en)
expect(not term_mismatch, f"命中词函数与计数函数对全量标签×两个方向恒一致: {term_mismatch[:3]}")
bh = label_tokens("balloon_hotair")
expect(label_tokens_matched_terms(bh, "outdoor, balloon_hotair, balloon") == ["balloon_hotair"],
       "balloon_hotair 只命中自己的牌,同牌上的 balloon 不算(3 (2) 装机同款)")

# ---- 批142:标签通道匹配级穷举 + 单元面(与 Swift 断言同集) ----
# 第四十七批的两条穷举只断言「词集里有这个词」;牌能不能匹配上是另一段逻辑
# (复合词自批126 起要求全部片段在词集)。Swift 侧探针实测:208 个复合键里
# 203 个中文搜不上自己的牌、5 个连英文直搜都搜不上、25 个同名多键的中文名
# 只能搜到先到先得的那一个。两侧同步升级为匹配级
fail_match_zh = [(cn, en) for en, cn in ZH.items()
                 if label_tokens_match_count(label_tokens(cn), en) == 0]
expect(not fail_match_zh, f"中文搜不到自己的标签牌: {fail_match_zh[:5]}")
fail_match_en = [en for en in ZH if label_tokens_match_count(label_tokens(en), en) == 0]
expect(not fail_match_en, f"英文直搜匹配不上自己的牌: {fail_match_en[:5]}")
print(f"匹配级穷尽: 中文 {len(ZH) - len(fail_match_zh)}/{len(ZH)}  "
      f"英文 {len(ZH) - len(fail_match_en)}/{len(ZH)}")
adult_cat = label_tokens("成年猫")
expect("adult_cat" in adult_cat and "adult" not in adult_cat,
       "「成年猫」词集=整键 adult_cat,不注入片段 adult")
expect(label_tokens_match_count(adult_cat, "adult") == 0,
       "「成年猫」不命中简单词牌 adult(全部人像照不灌入)")
expect(label_tokens_match_count(adult_cat, "person, adult, adult_cat") == 1,
       "「成年猫」命中自己那张牌上的 adult_cat,同牌的 person/adult 不计数")
expect(label_tokens_match_count(label_tokens("热气球"), "outdoor, balloon_hotair") == 1,
       "「热气球」命中 balloon_hotair 牌")
expect(label_tokens_match_count(label_tokens("身份证"), "play_card") == 0,
       "277 事故守卫(批126)不破:「身份证」词集含 card,仍不命中 play_card")
expect(label_tokens_match_count({"card"}, "play_card, credit_card") == 0,
       "词集只有片段 card 时任何 *_card 牌都不命中")
expect(label_tokens_match_count({"blue"}, "blue_sky") == 0,
       "词集只有片段 blue 时不命中 blue_sky(整键不在词集,仍走片段全中规则)")
expect(label_tokens_match_count(label_tokens("laundry_machine"), "laundry_machine") == 1,
       "laundry_machine 英文直搜命中自己的牌(machine 被当泛化词剔掉后靠整键)")
expect(label_tokens_match_count(label_tokens("天空"), "blue_sky") == 1
       and label_tokens_match_count(label_tokens("文档"), "printed_page") == 1,
       "复合键桥接生效:天空→blue_sky、文档→printed_page")
expect(label_tokens_match_count(label_tokens("电子产品"), "consumer_electronics") == 1
       and label_tokens_match_count(label_tokens("木头"), "wood_processed") == 1,
       "复合键桥接生效:电子产品→consumer_electronics、木头→wood_processed")
billboard = label_tokens("广告牌")
expect("billboard" in billboard and "billboards" in billboard,
       "「广告牌」两个键都在词集(此前先到先得只留一个)")
phone = label_tokens("手机")
expect("cell_phone" in phone and "mobile_phone" in phone, "「手机」两个键都在词集")
drone = label_tokens("无人机")
expect(label_tokens_match_count(drone, "drone") == 1
       and label_tokens_match_count(drone, "drone_machine") == 1,
       "「无人机」两张牌(drone / drone_machine)都命中")
expect(EN_KEYS_FROM_ZH.get("手机") == ["cell_phone", "mobile_phone"]
       and EN_KEYS_FROM_ZH.get("猫") == ["cat"],
       "反查全集:手机=两键且有序;单键名只有一个键")
# 批142 二修(验收 medium):多键名在短语里也要补全;盖不满/嵌套守卫
drone_show = label_tokens("无人机表演")
expect("drone" in drone_show and "drone_machine" in drone_show,
       "短语里的多键名也补全:「无人机表演」两个键都在词集")
billboard_design = label_tokens("广告牌设计")
expect("billboard" in billboard_design and "billboards" in billboard_design,
       "短语里的多键名也补全:「广告牌设计」两个键都在词集")
hybrid_billboard = label_tokens("华研广告牌")
expect("billboard" not in hybrid_billboard and "billboards" not in hybrid_billboard,
       "盖不满的原子词不捐:「华研广告牌」(专名+多键名)两个键都不进词集")
birthday_cake = label_tokens("生日蛋糕")
expect("cake_regular" not in birthday_cake,
       "嵌套只留最长:「生日蛋糕」里的「蛋糕」不单独捐,不带入 cake_regular")
multi_names = sorted(cn for cn, keys in EN_KEYS_FROM_ZH.items() if len(keys) >= 2)
print(f"反查全集: {len(EN_KEYS_FROM_ZH)} 个中文名,其中多键 {len(multi_names)} 个")

print("PASS: 镜像全绿(词典", len(DICTIONARY), "条/标签", len(ZH), "词/泛化", len(GENERIC), "个)")
