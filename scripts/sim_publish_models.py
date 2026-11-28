#!/usr/bin/env python3
"""
export_models.yml「发布模型包到 GitHub Release」那一步的本机模拟(批162,2026-11-28)。

用假的 gh 跑同一个 shell 脚本(直接从工作流文件里取 run: 块,不是另抄一份),
按序号注入「第 N 次写操作失败 / 第 N 次读操作失败」,核对不变量:
  I1 上传只碰临时名(*.staging),从不直接写正式名(首次发布除外)
  I2 核对(字节数 + sha256)通过之前,正式名资产一个都不动
  I3 任何时刻,四个文件每一个的内容都在线上(正式名或 *.previous)
  I4 旧包(*.previous)只在新包已经顶上正式名之后才删
  I5 清单最后换:manifest.json 的换名在三个包都换完之后
  I6 成功收尾 = 四个正式名都在、都是新一代、没有临时名残留
  I7 换名阶段失败后:原样重跑拒绝执行、什么都不改;手工改名修复后重跑能回到 I6
  I8 从不删除正式名资产(只改名);删除只认临时名
用法: python scripts/sim_publish_models.py [--quick]
      需要 bash 与 PyYAML;Windows 上自动找 Git Bash
"""
import os
import shutil
import subprocess
import sys
import tempfile

try:
    import yaml
except ImportError:
    sys.exit("需要 PyYAML: pip install pyyaml")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKFLOW = os.path.join(ROOT, ".github", "workflows", "export_models.yml")
STEP_PREFIX = "发布模型包到 GitHub Release"
FORMAL = ["models.zip", "face-only.zip", "models-so400m.zip", "manifest.json"]
PKGS = FORMAL[:3]
STAGE, OLD = ".staging", ".previous"

# 假 gh:只认发布步骤用到的几条命令,行为按 gh 的实际语义——
#   upload 不带 --clobber:同名资产已存在就报错,不覆盖
#   upload 带 --clobber:每个文件先删同名旧资产、再上传(gh 的 uploadWithDelete)
# 状态全在 $FAKE_GH_STATE:assets.tsv(id 名字 字节数 sha256 代次)、trace.log、
# snap/NNNN.tsv(每次写操作之后的线上全貌)。
FAKE_GH = r'''#!/usr/bin/env bash
set -euo pipefail
ST="${FAKE_GH_STATE:?}"
A="$ST/assets.tsv"; TR="$ST/trace.log"; SN="$ST/snap"
mkdir -p "$SN"
[ -f "$A" ] || : > "$A"
[ -f "$TR" ] || : > "$TR"
[ -f "$ST/seq" ] || echo 0 > "$ST/seq"
[ -f "$ST/gets" ] || echo 0 > "$ST/gets"
[ -f "$ST/id" ] || echo 1000 > "$ST/id"
die() { echo "gh(fake): $*" >&2; exit 1; }
count() { cat "$ST/$1"; }
bump() { echo $(( $(count "$1") + 1 )) > "$ST/$1"; }
nextid() { bump id; count id; }
field() { printf '%s\n' "$1" | awk -F'\t' -v c="$2" '{print $c}'; }
find_by() { awk -F'\t' -v c="$1" -v v="$2" '$c == v' "$A"; }
remove_id() { awk -F'\t' -v v="$1" '$1 != v' "$A" > "$A.tmp" && mv "$A.tmp" "$A"; }
sha_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print "sha256:" $1}'
  else shasum -a 256 "$1" | awk '{print "sha256:" $1}'; fi
}
snapshot() { cp "$A" "$SN/$(printf '%04d' "$(count seq)").tsv"; }
# 每次写操作:先记序号(故障注入按序号);被注入失败的不算「做过」,只记到 failed.log;
# 做了的记进 trace.log,并留一张快照
begin() {
  bump seq
  if [ "${FAKE_FAIL_SEQ:-0}" = "$(count seq)" ]; then
    echo "$1 $2" >> "$ST/failed.log"
    echo "gh(fake): write #$(count seq) ($1 $2) failed as injected" >&2
    return 1
  fi
  printf '%s\t%s\t%s\t%s\n' "$(count seq)" "$1" "$2" "${FAKE_UPLOAD_GEN:-new}" >> "$TR"
}
add_asset() {
  local name size digest
  name="$(basename "$1")"
  if [ -n "$(find_by 2 "$name")" ]; then die "HTTP 422: $name already_exists"; fi
  size="$(wc -c < "$1" | tr -d '[:space:]')"
  digest="$(sha_of "$1")"
  if [ "${FAKE_CORRUPT_UPLOAD:-}" = "$name" ]; then size=$((size + 1)); digest="sha256:deadbeef"; fi
  if [ "${FAKE_NO_DIGEST:-0}" = "1" ]; then digest="-"; fi
  printf '%s\t%s\t%s\t%s\t%s\n' "$(nextid)" "$name" "$size" "$digest" "${FAKE_UPLOAD_GEN:-new}" >> "$A"
}
upload_file() {
  local name row
  name="$(basename "$1")"
  row="$(find_by 2 "$name")"
  if [ -n "$row" ] && [ "$2" = "1" ]; then
    begin delete "$name" || return 1
    remove_id "$(field "$row" 1)"
    snapshot
  fi
  begin upload "$name" || return 1
  add_asset "$1"
  snapshot
}
cmd_api() {
  local method=GET ep="" newname="" id row old
  while [ $# -gt 0 ]; do
    case "$1" in
      -X|--method) method="$2"; shift 2 ;;
      -f|--raw-field) newname="${2#name=}"; shift 2 ;;
      -H|--header) shift 2 ;;
      *) ep="$1"; shift ;;
    esac
  done
  case "$method:$ep" in
    GET:repos/*/releases/tags/*)
      bump gets
      if [ "${FAKE_FAIL_GET:-0}" = "$(count gets)" ]; then die "read #$(count gets) failed as injected"; fi
      [ -e "$ST/release" ] || die "HTTP 404: release not found"
      awk -F'\t' 'BEGIN { printf "{\"id\": 1, \"tag_name\": \"models\", \"assets\": [" }
        NF >= 5 { if (n++) printf ", ";
                   d = ($4 == "" || $4 == "-") ? "null" : "\"" $4 "\"";
                   printf "{\"id\": %s, \"name\": \"%s\", \"size\": %s, \"state\": \"uploaded\", \"digest\": %s}", $1, $2, $3, d }
        END { print " ]}" }' "$A"
      ;;
    DELETE:repos/*/releases/assets/*)
      id="${ep##*/}"
      row="$(find_by 1 "$id")"
      [ -n "$row" ] || die "HTTP 404: asset $id not found"
      begin delete "$(field "$row" 2)" || return 1
      remove_id "$id"
      snapshot
      ;;
    PATCH:repos/*/releases/assets/*)
      id="${ep##*/}"
      row="$(find_by 1 "$id")"
      [ -n "$row" ] || die "HTTP 404: asset $id not found"
      old="$(field "$row" 2)"
      [ -z "$(find_by 2 "$newname")" ] || die "HTTP 422: name $newname already taken"
      begin rename "$old -> $newname" || return 1
      awk -F'\t' -v v="$id" -v n="$newname" 'BEGIN { OFS = "\t" } $1 == v { $2 = n } { print }' "$A" > "$A.tmp" && mv "$A.tmp" "$A"
      snapshot
      ;;
    *) die "unknown: $method $ep" ;;
  esac
}
# --repo X / -R X / --repo=X:去掉(模拟只认一个仓库)
args=()
while [ $# -gt 0 ]; do
  case "$1" in
    --repo|-R) shift 2 ;;
    --repo=*|-R=*) shift ;;
    *) args+=("$1"); shift ;;
  esac
done
if [ ${#args[@]} -gt 0 ]; then set -- "${args[@]}"; else set --; fi
case "${1:-}" in
  --version|-v) echo "gh version 0.0.0-fake (local simulation)" ;;
  release)
    case "${2:-}" in
      view) [ -e "$ST/release" ] || die "release not found"; echo "models" ;;
      create)
        [ "$3" = "models" ] || die "only tag=models"
        shift 3
        : > "$A"; : > "$TR"
        touch "$ST/release"
        files=()
        while [ $# -gt 0 ]; do
          case "$1" in
            --title|--notes) shift 2; continue ;;
            -*) shift ;;
            *) files+=("$1"); shift ;;
          esac
        done
        for f in "${files[@]}"; do upload_file "$f" 0 || exit 1; done ;;
      upload)
        [ "$3" = "models" ] || die "only tag=models"
        [ -e "$ST/release" ] || die "release not found"
        shift 3
        # 开关可以写在文件名后面(真 gh 先读完开关再处理文件)
        clobber=0; files=()
        while [ $# -gt 0 ]; do
          case "$1" in
            --clobber) clobber=1 ;;
            -*) ;;
            *) files+=("$1") ;;
          esac
          shift
        done
        for f in "${files[@]}"; do upload_file "$f" "$clobber" || exit 1; done ;;
      *) die "unknown: release ${2:-}" ;;
    esac ;;
  api) shift; cmd_api "$@" ;;
  *) die "unknown: ${1:-}" ;;
esac
'''


def p(*parts):
    """给 MSYS bash 用的路径:统一成正斜杠"""
    return os.path.join(*parts).replace("\\", "/")


def find_bash():
    cands = []
    if os.environ.get("SIM_BASH"):
        cands.append(os.environ["SIM_BASH"])
    if os.name == "nt":
        for c in (r"C:\Program Files\Git\usr\bin\bash.exe",
                  r"C:\Program Files (x86)\Git\usr\bin\bash.exe"):
            cands.append(c)
        git = shutil.which("git")
        if git:
            cands.append(os.path.join(os.path.dirname(os.path.dirname(git)), "usr", "bin", "bash.exe"))
        # Windows 自带的 System32\bash.exe 是 WSL 启动器,里面的路径体系不一样,不找它
        for c in (shutil.which("bash"), shutil.which("git")):
            if c and "system32" not in c.lower():
                cands.append(c)
    else:
        cands.append(shutil.which("bash") or "bash")
    for c in cands:
        if not (os.path.isfile(c) or c == "bash"):
            continue
        r = subprocess.run([c, "-c", "echo ok-$((1+1))"], capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip() == "ok-2":
            return c
    sys.exit("找不到可用的 bash(Windows 上请装 Git for Windows,或用 SIM_BASH 指定 bash.exe)")


def write_files(d, tag, pad=0):
    for i, n in enumerate(FORMAL):
        with open(os.path.join(d, n), "wb") as f:
            f.write(("%s content of %s #%d\n" % (tag, n, i)).encode() + b"x" * pad)


def read_tsv(path):
    rows = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 5:
                    rows.append({"id": parts[0], "name": parts[1], "size": parts[2],
                                 "digest": parts[3], "gen": parts[4]})
    return rows


def names(rows):
    return [r["name"] for r in rows]


def sig(rows):
    return tuple(sorted((r["id"], r["name"]) for r in rows))


def formal_sig(rows):
    """四个正式名资产的签名(临时名资产留在网上不算差异)"""
    return tuple(sorted((r["id"], r["name"]) for r in rows if r["name"] in FORMAL))


def content_ok(rows):
    have = set(names(rows))
    return all(f in have or f + OLD in have for f in FORMAL)


def clean(rows, gen):
    return sorted(names(rows)) == sorted(FORMAL) and all(r["gen"] == gen for r in rows)


def touches_formal(trace):
    """动到正式名资产的写操作(删除正式名,或把正式名改走)"""
    out = []
    for seq, op, name, _ in trace:
        if op == "delete" and name in FORMAL:
            out.append((seq, op, name))
        elif op == "rename":
            old, new = name.split(" -> ")
            if old in FORMAL:
                out.append((seq, op, name))
    return out


def out(r):
    return (r.stdout + r.stderr).strip()[-700:]


class World:
    """一个临时的「线上 Release + 工作目录」。每次 run() 的轨迹、序号、快照单独归档,
    trace()/snaps()/count() 只看最近一次运行。"""

    def __init__(self, tmp, fake_dir, bash, with_release=True, old_pad=0):
        self.bash, self.fake_dir = bash, fake_dir
        self.root = tempfile.mkdtemp(dir=tmp)
        self.st = p(self.root, "state")
        self.work = p(self.root, "work")
        self.runs = 0
        os.makedirs(self.st)
        os.makedirs(self.work)
        if with_release:
            old = p(self.root, "oldwork")
            os.makedirs(old)
            write_files(old, "OLD", pad=old_pad)   # old_pad>0:线上的包比这次导出的大
            r = self.gh("release", "create", "models", *[p(old, n) for n in FORMAL], gen="old")
            assert r.returncode == 0, r.stderr

    def _start(self):
        """新一次运行开始:把上一次的轨迹归档,记下开跑前的线上全貌"""
        self.runs += 1
        hist = os.path.join(self.st, "hist", str(self.runs))
        os.makedirs(hist)
        for c in ("trace.log", "failed.log", "seq", "gets", "snap"):
            src = os.path.join(self.st, c)
            if os.path.exists(src):
                shutil.move(src, os.path.join(hist, c))
        for c in ("trace.log", "failed.log"):
            with open(os.path.join(self.st, c), "w"):
                pass
        for c in ("seq", "gets"):
            with open(os.path.join(self.st, c), "w") as f:
                f.write("0")
        os.makedirs(os.path.join(self.st, "snap"))
        assets = os.path.join(self.st, "assets.tsv")
        if not os.path.exists(assets):      # 首次发布:线上还没有这个 Release
            with open(assets, "w"):
                pass
        shutil.copyfile(assets, os.path.join(self.st, "snap", "0000.tsv"))

    def env(self, gen, inj=None, fake_first=False):
        env = {k: v for k, v in os.environ.items() if not k.startswith("FAKE_")}
        env["FAKE_GH_STATE"] = self.st
        env["FAKE_UPLOAD_GEN"] = gen
        env["GITHUB_REPOSITORY"] = "owner/repo"
        env.update(inj or {})
        if fake_first:
            env["PATH"] = self.fake_dir + os.pathsep + env.get("PATH", "")
        return env

    def _call(self, args, cwd, env):
        return subprocess.run([self.bash, *args], cwd=cwd, env=env,
                              capture_output=True, text=True, encoding="utf-8", errors="replace")

    def gh(self, *args, gen="junk", inj=None):
        """直接调假 gh(搭初始状态、模拟手工修复)"""
        return self._call([p(self.fake_dir, "gh"), *args], None, self.env(gen, inj))

    def run(self, script, inj=None, gen="new"):
        self._start()
        write_files(self.work, gen.upper())
        # GitHub 跑没写 shell: 的 run 步骤用的就是 bash -e(不带 pipefail、-u)
        return self._call(["-e", p(script)], self.work, self.env(gen, inj, fake_first=True))

    def rename(self, src, dst):
        """模拟手工修复:把 src 改名为 dst"""
        row = next(r for r in self.rows() if r["name"] == src)
        r = self.gh("api", "-X", "PATCH", "-f", "name=" + dst,
                    "repos/owner/repo/releases/assets/" + row["id"], gen=row["gen"])
        assert r.returncode == 0, r.stderr

    def rows(self):
        return read_tsv(os.path.join(self.st, "assets.tsv"))

    def trace(self):
        """最近一次运行的写操作:(序号, 操作, 对象, 代次)"""
        with open(os.path.join(self.st, "trace.log"), encoding="utf-8") as f:
            return [(int(a), op, name, gen)
                    for a, op, name, gen in (line.rstrip("\n").split("\t") for line in f if line.strip())]

    def snaps(self):
        d = os.path.join(self.st, "snap")
        return [(int(n[:4]), read_tsv(os.path.join(d, n))) for n in sorted(os.listdir(d))]

    def count(self, c):
        with open(os.path.join(self.st, c)) as f:
            return int(f.read().strip() or 0)


class Report:
    def __init__(self):
        self.n = 0
        self.bad = []

    def check(self, ok, name, detail=""):
        self.n += 1
        if not ok:
            self.bad.append((name, detail))


def old_deleted_after_new(w):
    """I4:每次删 X.previous 之前的最后一张快照里,X 已经是新一代"""
    snaps = w.snaps()
    for seq, op, name, _ in w.trace():
        if op != "delete" or not name.endswith(OLD):
            continue
        before = [rows for s, rows in snaps if s < seq]
        x = name[:-len(OLD)]
        if not before or not any(r["name"] == x and r["gen"] == "new" for r in before[-1]):
            return False
    return True


def after_swap_failure(w, script, rep, label):
    """I7:失败之后先原样重跑;若线上有正式名缺失,重跑必须拒绝;手工修复后再重跑应成功"""
    broken = [f for f in FORMAL if f not in names(w.rows())]
    before = sig(w.rows())
    r2 = w.run(script)
    if broken:
        rep.check(r2.returncode != 0 and sig(w.rows()) == before and not w.trace(),
                  "I7a " + label + ": 原样重跑拒绝执行、什么都没改", out(r2))
        for f in broken:
            src = f + STAGE if f + STAGE in names(w.rows()) else f + OLD
            w.rename(src, f)
        r3 = w.run(script)
        rep.check(r3.returncode == 0 and clean(w.rows(), "new"),
                  "I7b " + label + ": 手工修复后重跑成功收尾", out(r3))
    else:
        rep.check(r2.returncode == 0 and clean(w.rows(), "new"),
                  "I7  " + label + ": 没有正式名缺失,重跑成功收尾", out(r2))


def suite(script, fake_dir, bash, tmp, quick):
    rep = Report()

    def world(**kw):
        return World(tmp, fake_dir, bash, **kw)

    # S1 正常重发
    w = world()
    r = w.run(script)
    rep.check(r.returncode == 0, "S1 正常重发:退出码 0", out(r))
    rep.check(clean(w.rows(), "new"), "I6 正常重发收尾:四个正式名、都是新一代、无临时名",
              str(names(w.rows())))
    tr = w.trace()
    ups = [t for t in tr if t[1] == "upload"]
    rep.check(len(ups) == 4 and all(t[2].endswith(STAGE) for t in ups), "I1 四次上传全是临时名", str(ups))
    ren = [t for t in tr if t[1] == "rename"]
    seq_to = lambda n: [t[0] for t in ren if t[2].endswith("-> " + n)]
    m, ps = seq_to("manifest.json"), [seq_to(x) for x in PKGS]
    rep.check(bool(m) and all(ps) and m[0] > max(x[0] for x in ps), "I5 清单最后换", str(ren))
    snaps = w.snaps()
    misses = [(s, f) for s, rows in snaps for f in FORMAL if f not in names(rows)]
    rep.check(len(misses) == 4, "I3 正式名空缺只有 4 个瞬间(每个文件换名一次)", str(misses))
    rep.check(all(content_ok(rows) for s, rows in snaps), "I3 任何时刻四个文件的内容都在线上")
    rep.check(old_deleted_after_new(w), "I4 旧包只在新包顶上正式名之后才删")
    rep.check(not any(t[1] == "delete" and t[2] in FORMAL for t in tr), "I8 从不删除正式名资产")
    swap_seqs = [t[0] for t in tr if t[1] in ("rename", "delete")]
    gets = w.count("gets")
    n_writes = len(swap_seqs)
    if quick and len(swap_seqs) > 4:
        swap_seqs = [swap_seqs[0], swap_seqs[len(swap_seqs) // 2 - 1], swap_seqs[len(swap_seqs) // 2], swap_seqs[-1]]
    if quick and gets > 10:
        gets = 10

    # S2 上传中途失败(第 1~4 次写操作)
    for k in range(1, 5):
        w = world()
        base = formal_sig(w.rows())
        r = w.run(script, {"FAKE_FAIL_SEQ": str(k)})
        rep.check(r.returncode != 0, "S2 上传失败(写 #%d):非零退出" % k, out(r))
        rep.check(not touches_formal(w.trace()), "I2 上传失败(写 #%d):正式名一个没动" % k)
        rep.check(formal_sig(w.rows()) == base, "S2 上传失败(写 #%d):线上与之前一模一样" % k, str(names(w.rows())))
        rep.check(all(content_ok(rows) for s, rows in w.snaps()), "I3 上传失败(写 #%d):内容始终在线" % k)

    # S3 核对不过(某个临时资产传上去与本地不符)
    for n in FORMAL:
        w = world()
        base = formal_sig(w.rows())
        r = w.run(script, {"FAKE_CORRUPT_UPLOAD": n + STAGE})
        rep.check(r.returncode != 0 and not touches_formal(w.trace()) and formal_sig(w.rows()) == base,
                  "I2 核对不过(%s):中止,正式名一个没动" % n, out(r))

    # S3b 体量检查:新包不到线上同名包的一半(导出残缺 / 拿错文件)就中止,正式名一个没动
    for pad, label in ((20000, "新包只有线上的千分之一"), (40, "新包略低于一半线")):
        w = world(old_pad=pad)
        base = formal_sig(w.rows())
        r = w.run(script)
        rep.check(r.returncode != 0 and not touches_formal(w.trace()) and formal_sig(w.rows()) == base,
                  "I2 体量检查(%s):中止,正式名一个没动" % label, out(r))
    w = world(old_pad=20)
    r = w.run(script)
    rep.check(r.returncode == 0 and clean(w.rows(), "new"), "S3b 体量检查(新包略高于一半线):照常发布", out(r))

    # S4 接口不返回 sha256:只核对字节数,照常发布
    w = world()
    r = w.run(script, {"FAKE_NO_DIGEST": "1"})
    rep.check(r.returncode == 0 and clean(w.rows(), "new"), "S4 接口没有 sha256:照常发布并收尾", out(r))

    # S5 换名阶段每一步失败
    for k in swap_seqs:
        w = world()
        r = w.run(script, {"FAKE_FAIL_SEQ": str(k)})
        rep.check(r.returncode != 0, "S5 换名失败(写 #%d):非零退出" % k, out(r))
        rep.check(all(content_ok(rows) for s, rows in w.snaps()), "I3 换名失败(写 #%d):内容始终在线" % k)
        rep.check(old_deleted_after_new(w), "I4 换名失败(写 #%d):旧包没被提前删" % k)
        rep.check(not any(t[1] == "delete" and t[2] in FORMAL for t in w.trace()),
                  "I8 换名失败(写 #%d):没有删除正式名资产" % k)
        after_swap_failure(w, script, rep, "换名失败(写 #%d)" % k)

    # S6 每一次读操作失败
    for g in range(1, gets + 1):
        w = world()
        base = formal_sig(w.rows())
        r = w.run(script, {"FAKE_FAIL_GET": str(g)})
        rep.check(all(content_ok(rows) for s, rows in w.snaps()), "I3 读失败(读 #%d):内容始终在线" % g)
        if touches_formal(w.trace()):
            after_swap_failure(w, script, rep, "读失败(读 #%d)" % g)
        else:
            rep.check(r.returncode != 0 and formal_sig(w.rows()) == base,
                      "I2 读失败(读 #%d,换名前):非零退出,正式名一个没动" % g, out(r))

    # S7 首次发布:线上还没有这个 Release
    w = world(with_release=False)
    r = w.run(script)
    rep.check(r.returncode == 0 and clean(w.rows(), "new"), "S7 首次发布:四个正式名直接建好", out(r))

    # S8 上次失败留下的临时名资产:清掉后照常发布
    w = world()
    junk = p(w.root, "junk")
    os.makedirs(junk)
    stale = ["models.zip" + STAGE, "manifest.json" + OLD]
    for n in stale:
        with open(os.path.join(junk, n), "wb") as f:
            f.write(b"stale junk\n")
    r = w.gh("release", "upload", "models", *[p(junk, n) for n in stale])
    assert r.returncode == 0, r.stderr
    r = w.run(script)
    rep.check(r.returncode == 0 and clean(w.rows(), "new"), "S8 清掉残留的临时名资产后照常发布", out(r))

    # S9 线上本来就缺一个正式名(上次换名中断):什么都不改
    w = world()
    victim = next(x for x in w.rows() if x["name"] == "face-only.zip")
    r = w.gh("api", "-X", "DELETE", "repos/owner/repo/releases/assets/" + victim["id"], gen="old")
    assert r.returncode == 0, r.stderr
    before = sig(w.rows())
    r = w.run(script)
    rep.check(r.returncode != 0 and sig(w.rows()) == before and not w.trace(),
              "I7a 线上缺正式名:拒绝发布、什么都没改", out(r))
    return rep, n_writes, gets


def main():
    quick = "--quick" in sys.argv
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    with open(WORKFLOW, encoding="utf-8") as f:
        wf = yaml.safe_load(f)
    step = next(s for s in wf["jobs"]["export"]["steps"] if s.get("name", "").startswith(STEP_PREFIX))
    assert "shell" not in step, "发布步骤若指定了 shell,这里的模拟要跟着改"
    tmp = tempfile.mkdtemp(prefix="sim_publish_")
    try:
        bash = find_bash()
        fake_dir = os.path.join(tmp, "fakebin")
        os.makedirs(fake_dir)
        gh_path = os.path.join(fake_dir, "gh")
        with open(gh_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(FAKE_GH)
        try:
            os.chmod(gh_path, 0o755)
        except OSError:
            pass
        script = os.path.join(tmp, "publish_step.sh")
        with open(script, "w", encoding="utf-8", newline="\n") as f:
            f.write(step["run"])
        r = subprocess.run([bash, "-n", p(script)], capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        if r.returncode != 0:
            print("bash -n 不过:\n" + r.stderr)
            return 1
        rep, n_writes, gets = suite(script, fake_dir, bash, tmp, quick)
        print("换名阶段的写操作 %d 个、读操作 %d 个%s,逐个注入失败都跑过;共 %d 项核对"
              % (n_writes, gets, "(--quick 抽查)" if quick else "", rep.n))
        for name, detail in rep.bad:
            print("[FAIL] " + name + ("\n       " + detail.replace("\n", "\n       ") if detail else ""))
        print("%s %d / %d" % ("PASS" if not rep.bad else "FAIL", rep.n - len(rep.bad), rep.n))
        return 1 if rep.bad else 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
