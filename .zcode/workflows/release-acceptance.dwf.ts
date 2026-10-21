/* zcode-workflow
description: MediaSeek TestFlight 发版前的自动验收:核对仓库边界(未推送/未提交)、四项 CI
  结论(区分苹果容量拒单与真失败、路径不适用)、逐提交评审员核对提交声明与项目教训清单、独立复核发现,产出可发版/暂缓/不可发版三态验收报告。
whenToUse: 用户说「跑验收/发版前验收/release acceptance」时;每次 gh workflow run release.yml 之前作为质量门禁运行。
*/
interface Commit {
  /** 完整提交哈希 */
  sha: string;
  /** 提交说明首行 */
  subject: string;
}
interface CiCheck {
  /** 工作流名 */
  name: string;
  /** success=绿;real-failure=真失败;capacity-failure=容量拒单;running=排队中;none=缺失且该提交触及触发路径(未决);skip-ok=缺失但提交未触及触发路径(不适用) */
  state: "success" | "real-failure" | "capacity-failure" | "running" | "none" | "skip-ok";
}
interface Finding {
  /** 文件:行号,如 MediaSeek/Core/SearchEngine.swift:728 */
  where: string;
  /** 一句话:什么问题 */
  what: string;
  /** 证据:看到的代码行或命令输出 */
  evidence: string;
  /** verified=复核员独立复现;unconfirmed=复核未通过;resolved-by-later-commit=后续提交已修(跨提交抵消) */
  status: "verified" | "unconfirmed" | "resolved-by-later-commit";
  /** high=会崩溃或数据错;medium=功能受损;low=风格 */
  severity: "low" | "medium" | "high";
  /** 发现该问题的提交哈希(前 7 位) */
  sourceSha?: string;
}
interface ReviewResult {
  sha: string;
  subject: string;
  findings: Finding[];
}
interface Confirmation {
  /** 复核员能否独立复现该问题 */
  holds: boolean;
  /** 一句话:复现依据或不成立原因 */
  note: string;
}

const REPO = "yangru158158-ux/MediaSeek";
const PS = "$env:GH_TOKEN = (Get-Content C:\\Users\\44957\\.zcode\\workspace\\default\\.gh_token -Raw).Trim(); & 'C:\\Program Files\\GitHub CLI\\gh.exe' ";
const REQUIRED_CI = [
  "Build Check",
  "Build Unsigned IPA",
  "Face Recovery and Align Tests",
  "Query Understanding Tests",
];

artifact.table("checks", {
  title: "CI 检查结论",
  key: "name",
  columns: [
    { field: "name", label: "检查" },
    { field: "state", label: "结论" },
  ],
});

phase("盘点仓库与上次发版边界");
const status = await world.run("git", ["-C", "MediaSeek", "status", "--porcelain"]);
const dirty = status.stdout.trim();
const head = await world.run("git", ["-C", "MediaSeek", "rev-parse", "HEAD"]);
const headSha = head.stdout.trim();
const unpushedRun = await world.run("git", ["-C", "MediaSeek", "rev-list", "origin/main..HEAD", "--count"]);
const unpushed = Number(unpushedRun.stdout.trim() || "0");
const relRun = await world.run(
  "powershell",
  ["-Command", PS + "run list -R " + REPO + " --workflow release.yml --status success --limit 1 --json headSha"],
  { timeoutMs: 120000 },
);
const rel = (JSON.parse(relRun.stdout || "[]") as { headSha?: string }[])[0];
const lastSha = rel?.headSha ?? "";
const logRun = await world.run(
  "git",
  ["-C", "MediaSeek", "log", "--format=%H %s", ...(lastSha ? [lastSha + "..HEAD"] : ["-n", "5"])],
);
const commits: Commit[] = logRun.stdout
  .trim()
  .split("\n")
  .filter(Boolean)
  .map((l) => ({ sha: l.slice(0, 40), subject: l.slice(41).trim() }));
log(
  `待发 ${headSha.slice(0, 7)}:自上次发版(${lastSha.slice(0, 7) || "无"})以来 ${commits.length} 个提交;` +
    `工作区${dirty ? "有未提交改动" : "干净"};未推送 ${unpushed} 个`,
);

const diffRun = await world.run(
  "git",
  ["-C", "MediaSeek", "diff", "--name-only", ...(lastSha ? [lastSha] : ["HEAD~5"]), "HEAD"],
);
const changed = diffRun.stdout.trim().split("\n").filter(Boolean);
// 各工作流的触发路径(与 .github/workflows/*.yml 的 paths 一致)
const TRIGGER_PATHS: Record<string, string[]> = {
  "Face Recovery and Align Tests": ["MediaSeek/Core/FaceKit", "MediaSeek/Core/FaceRecoveryPolicy", "MediaSeek/Views/PeopleView", "scripts/face_"],
  "Query Understanding Tests": ["MediaSeek/Core/QueryUnderstanding", "MediaSeek/Core/LabelZh", "scripts/query_"],
};
const touches = (name: string) =>
  (TRIGGER_PATHS[name] ?? []).some((p) => changed.some((f) => f.startsWith(p) || f.includes(p.slice(1))));

phase("核对 CI 检查结论");
const ciRun = await world.run(
  "powershell",
  ["-Command", PS + "run list -R " + REPO + " --commit " + headSha + " --json databaseId,name,conclusion,status"],
  { timeoutMs: 120000 },
);
const allRuns = JSON.parse(ciRun.stdout || "[]") as {
  databaseId: number;
  name: string;
  conclusion: string;
  status: string;
}[];
const checks: CiCheck[] = [];
for (const name of REQUIRED_CI) {
  const r = allRuns.find((x) => x.name === name);
  if (!r) {
    checks.push({ name, state: touches(name) ? "none" : "skip-ok" });
  } else if (r.status !== "completed") {
    checks.push({ name, state: "running" });
  } else if (r.conclusion === "success") {
    checks.push({ name, state: "success" });
  } else {
    // 区分真失败与苹果容量拒单(15 分钟无人接单,与代码无关,重跑即可)
    const view = await world.run(
      "powershell",
      ["-Command", PS + "run view " + r.databaseId + " -R " + REPO],
      { timeoutMs: 120000 },
    );
    const text = view.stdout + view.stderr;
    checks.push({
      name,
      state: text.includes("not acquired by Runner") ? "capacity-failure" : "real-failure",
    });
  }
  report({ name, state: checks[checks.length - 1]?.state ?? "none" }, "checks");
}
log(`CI:${checks.map((c) => c.name + "=" + c.state).join("; ")}`);

phase("逐个核对发版改动");
const toReview = commits.slice(0, 6);
const reviewed: ReviewResult[] =
  toReview.length === 0
    ? []
    : await Promise.all(
        toReview.map((c) =>
          agent(`评审员-${c.sha.slice(0, 7)}`, {
            system:
              "你是 iOS App「智搜」发版验收的代码评审员,在仓库 MediaSeek 里工作(git 命令要用 git -C MediaSeek ...)。" +
              "只读不改:严禁编辑任何文件,只允许读文件和跑只读命令。" +
              "若某条声明无法核验(证据不足),如实报'无法核验',绝不编造;没有问题就返回空 findings,不为交差硬找。",
          }).ask<ReviewResult>(
            `核对提交 ${c.sha}(说明:${c.subject})。` +
              `先运行 git -C MediaSeek show ${c.sha} 看完整 diff,再打开涉及的文件读上下文,对照提交说明逐条核验。` +
              "项目历史教训清单(逐项过):\n" +
              "① 提交说明的每条声明须与 diff 相符:算法盐(如 aN→aN+1)是否两处同步(PeopleView 里 start 与 hydrate 两个函数)、" +
              "CI 断言数声明与 scripts/*_test.swift 里实际 expect 数及 print 数字是否一致;\n" +
              "② 改了函数返回类型/签名必须全查调用点与 early return(历史上漏改 return [] 红过两次 CI);\n" +
              "③ 归组/识别逻辑改动必须升盐,否则旧快照/缓存复活旧判决;\n" +
              "④ 不得残留调试代码(print 调试输出、TODO 临时代码、误提交的缓存目录如 .mimosa);\n" +
              "⑤ 新增 Swift 文件在 MediaSeek/ 目录下会自动进工程,无需手动登记——若提交说明声称手动登记过,指出矛盾。\n" +
              "返回 ReviewResult:findings 每条带 where(文件:行)/what(一句话)/evidence(你看到的行)/severity" +
              "(high=会崩或数据错,medium=功能受损,low=风格),status 统一填 unconfirmed(后续有人复核)。sha/subject 原样带回。",
          ),
        ),
      );
for (const r of reviewed) {
  log(`提交 ${r.sha.slice(0, 7)} 核对完成:${r.findings.length} 个发现`);
}

phase("独立复核发现的问题");
// 带来源提交:评审员只看自己那个提交的 diff,问题可能在后续提交里被修
const rawFindings: Finding[] = reviewed.flatMap((r) =>
  r.findings.map((f) => ({ ...f, sourceSha: r.sha.slice(0, 7) })),
);
const confirmations: Confirmation[] = await Promise.all(
  rawFindings.map((f, i) =>
    agent(`复核员-${i}`, {
      system:
        "你是独立复核员:只凭自己的眼睛重新查证别人报的问题,不看别人的推理过程。" +
        "只读不改:严禁编辑任何文件。无法复现就说不成立,不要迁就。",
    }).ask<Confirmation>(
      `复核这个发版阻断候选发现(在仓库 MediaSeek 里,git 命令用 git -C MediaSeek ...):\n${JSON.stringify(f)}\n` +
        "自己打开 where 指向的文件、读上下文、必要时跑只读命令(git show/diff/grep),判断问题是否真实存在。" +
        "返回 {holds: 是否成立, note: 一句话复现依据或不成立的原因}。",
    ),
  ),
);
const findings: Finding[] = rawFindings.map((f, i) => ({
  ...f,
  status: confirmations[i]?.holds ? "verified" : "unconfirmed",
}));

phase("跨提交抵消检查");
// 来源不是 HEAD 的发现:交给抵消检查员只读 HEAD 现态——后续提交可能已修
const headShort = headSha.slice(0, 7);
const fromEarlier = findings.filter((f) => f.status === "verified" && f.sourceSha !== undefined && f.sourceSha !== headShort);
const offsetResolutions = await Promise.all(
  fromEarlier.map((f, i) =>
    agent(`抵消检查员-${i}`, {
      system:
        "你是跨提交抵消检查员:一个问题在较早的提交里被发现,你要判断它在当前 HEAD 上是否已被后续提交修复。" +
        "只读不改:严禁编辑任何文件。判断依据是 HEAD 上的实际代码,不是 git log 或提交说明。",
    }).ask<{ stillPresent: boolean; note: string }>(
      `以下问题是在提交 ${f.sourceSha} 的 diff 中发现的:\n` +
      `位置:${f.where}\n问题:${f.what}\n证据:${f.evidence}\n\n` +
      `请打开 MediaSeek 仓库里该文件在当前 HEAD(${headShort})上的实际内容(git 命令用 git -C MediaSeek ...),` +
      `读 where 指向的位置及其上下文,判断:该问题在 HEAD 上是否仍然存在?` +
      `返回 {stillPresent: 是否仍存在, note: 一句话依据(如'已改为 try? prepare'或'仍是裸 try')}`,
    ),
  ),
);
const effectiveFindings: Finding[] = findings.map((f) => {
  const idx = fromEarlier.indexOf(f);
  if (idx >= 0 && offsetResolutions[idx] && !offsetResolutions[idx].stillPresent) {
    return { ...f, status: "resolved-by-later-commit" as const };
  }
  return f;
});
log(`跨提交抵消:${fromEarlier.length} 个非 HEAD 来源的发现中,` +
  `${fromEarlier.filter((_, i) => offsetResolutions[i] && !offsetResolutions[i].stillPresent).length} 个已被后续提交修复`);

phase("产出验收报告");
const realFail = checks.filter((c) => c.state === "real-failure");
const unsettled = checks.filter((c) => c.state === "capacity-failure" || c.state === "running" || c.state === "none");
const skipped = checks.filter((c) => c.state === "skip-ok");
const highs = effectiveFindings.filter((f) => f.severity === "high" && f.status === "verified");
const blockers = [
  ...realFail.map((c) => `CI 真失败:${c.name}`),
  ...(dirty ? [`工作区有未提交改动(不会随本次发版):${dirty.split("\n").slice(0, 3).join("; ")}`] : []),
  ...(unpushed > 0 ? [`${unpushed} 个提交未推送到 origin,发版会构建不到`] : []),
  ...highs.map((f) => `高危问题:${f.what}(${f.where})`),
];
const verdict =
  commits.length === 0
    ? "无需发版:自上次成功发版以来没有新提交"
    : blockers.length > 0
      ? `不可发版——${blockers.length} 项阻断`
      : unsettled.length > 0
        ? `暂缓发版——${unsettled.length} 项 CI 检查未得出结论(容量拒单/排队中/缺失),重跑 CI 后再验收`
        : "可以发版——CI 全绿,改动核对无已确认的高危问题";
await artifact.markdown(
  "acceptance-report",
  [
    "# 发版前验收报告",
    "",
    `**${verdict}**`,
    "",
    "## 基本面",
    `- 待发 HEAD:${headSha.slice(0, 7)};自上次发版(${lastSha.slice(0, 7) || "无"})以来 ${commits.length} 个提交`,
    `- 未推送 origin:${unpushed} 个;工作区:${dirty ? "有未提交改动" : "干净"}`,
    "",
    "## CI 检查",
    ...checks.map((c) => `- ${c.name}:**${c.state}**`),
    ...(unsettled.length
      ? ["", "> 容量拒单 = 苹果 macOS runner 15 分钟无人接单,与代码无关,重跑即可;真失败 = 编译/测试问题,必须修。"]
      : []),
    "",
    "## 逐提交核对",
    ...(reviewed.length === 0
      ? ["- 无待发提交"]
      : reviewed.flatMap((r) => [
          `### ${r.sha.slice(0, 7)} ${r.subject}`,
          ...(r.findings.length === 0 ? ["- 无问题"] : r.findings.map((f) => `- [${f.severity}] ${f.what}(${f.where})`)),
        ])),
    "",
    "## 全部发现(含复核与跨提交抵消)",
    ...(effectiveFindings.length === 0
      ? ["- 无"]
      : effectiveFindings.map((f) => {
          if (f.status === "resolved-by-later-commit") {
            return `- ~~[${f.severity}] ${f.what} — ${f.where}~~\n  ✅ 已被后续提交修复(来源:${f.sourceSha})`;
          }
          return `- [${f.severity}/${f.status}] ${f.what} — ${f.where}\n  证据:${f.evidence}`;
        })),
    "",
    "## 建议",
    verdict.startsWith("可以发版")
      ? "- 执行发版:gh workflow run release.yml(由主会话或用户触发,本工作流不代发)"
      : blockers.length > 0
        ? blockers.map((b) => `- 处理阻断项:${b}`)
        : unsettled.map((c) => `- 重跑未决检查:${c.name}(gh run rerun 或等容量恢复重新 push 触发)`),
  ].join("\n"),
  { title: "发版前验收报告", description: verdict, primary: true },
);

const finalReport = {
  conclusion:
    `${verdict}。待发 ${headSha.slice(0, 7)}(自上次发版 ${commits.length} 个提交,CI ` +
    `${checks.filter((c) => c.state === "success").length}/${checks.length} 绿),` +
    `改动核对 ${reviewed.length} 个提交、确认 ${effectiveFindings.filter((f) => f.status === "verified").length} 个问题` +
    `(高危 ${highs.length},已抵消 ${effectiveFindings.filter((f) => f.status === "resolved-by-later-commit").length})。`,
  findings: effectiveFindings,
  verified: [
    "git status/rev-parse/rev-list 确认仓库边界(确定性)",
    "gh run list --commit 确认四项 CI 结论,失败运行逐一 gh run view 区分容量拒单与真失败(确定性)",
    "每个待发提交由独立评审员读 diff 核对提交声明与项目教训清单",
    ...(rawFindings.length > 0 ? ["每个发现由独立复核员只读复现"] : []),
    ...(fromEarlier.length > 0 ? ["非 HEAD 来源的发现经跨提交抵消检查(只读 HEAD 现态验证是否已修)"] : []),
  ],
  notCovered: [
    ...(commits.length > 6 ? [`提交数 ${commits.length} 超过 6,只核对了前 6 个`] : []),
    "未运行 iOS 真机/模拟器实测(构建与策略测试由 CI 覆盖,运行时行为仍靠设备验收)",
    "release.yml 发版动作本身不在本验收范围内(验收通过后人来触发)",
  ],
};
return finalReport;