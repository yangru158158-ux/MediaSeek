#!/usr/bin/env python3
"""发版验收轨道:容量拒单自动重派,四项全绿后触发 release.yml。

GitHub 免费 macOS runner 高峰期会「拒单」:run conclusion=failure、
job conclusion=cancelled、steps 一个没跑("not acquired by runner")。
这既不是代码问题也不自愈——必须人工重派。本脚本把重派自动化:

  每 3 分钟一轮,四项(Build Check / IPA / Query Tests / Face Tests):
    - 拒单(0 步) → 立即 rerun-failed-jobs / 重新 dispatch
    - 真失败(跑了步骤但挂) → 停止,不重试不发包,退出码 2
    - 全绿 → 触发 release.yml(workflow_dispatch main),退出码 0
    - 90 分钟仍未全绿 → 放弃,退出码 3(把状态留给下一班)

用法: python scripts/ci_retry_orbit.py --max-min 90
"""
import argparse
import json
import subprocess
import sys
import time
import urllib.request

REPO = "yangru158158-ux/MediaSeek"
# 验收工作流:按 HEAD sha 找各自最新 run;face/query 按需手动补派
WANT = ["Build Check", "Build Unsigned IPA", "Query Understanding Tests",
        "Face Recovery and Align Tests", "Model Validation"]
# 缺席时自动补派的工作流(其余靠 push 路径自动触发)
MISSING_DISPATCH = {
    "Face Recovery and Align Tests": "face_test.yml",
    "Query Understanding Tests": "query_test.yml",
}
ACCEPT_SHA = None  # main HEAD(验收目标提交)


def token():
    out = subprocess.run(
        ["git", "credential", "fill"],
        input="protocol=https\nhost=github.com\n\n",
        capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.startswith("password="):
            return line.split("=", 1)[1]
    raise SystemExit("取不到 GitHub 令牌")


def api(path, tok, method="GET"):
    req = urllib.request.Request(
        f"https://api.github.com/repos/{REPO}/{path}",
        headers={"Authorization": f"token {tok}"}, method=method)
    with urllib.request.urlopen(req, timeout=30) as r:
        body = r.read()
        return r.status, (json.loads(body) if body else {})


def run_state(rid, tok):
    _, run = api(f"actions/runs/{rid}", tok)
    _, jobs = api(f"actions/runs/{rid}/jobs", tok)
    steps = sum(1 for j in jobs.get("jobs", [])
                for s in j.get("steps", []) if s["status"] == "completed")
    return {
        "name": run.get("name", str(rid)),
        "status": run.get("status"),
        "conclusion": run.get("conclusion"),
        "steps": steps,
    }


def latest_runs(tok, sha):
    """HEAD sha 上四个工作流各自的最新 run;缺哪个记 None。"""
    _, d = api(f"actions/runs?head_sha={sha}&per_page=50", tok)
    found = {}
    for r in d.get("workflow_runs", []):
        if r["name"] in WANT and r["name"] not in found:
            found[r["name"]] = r["id"]
    return {n: found.get(n) for n in WANT}


def is_rejected(st):
    return st["status"] == "completed" and st["conclusion"] == "failure" and st["steps"] == 0


def is_real_fail(st):
    return st["status"] == "completed" and st["conclusion"] != "success" and st["steps"] > 0


def is_green(st):
    return st["status"] == "completed" and st["conclusion"] == "success"


def rerun(rid, tok):
    code, _ = api(f"actions/runs/{rid}/rerun-failed-jobs", tok, method="POST")
    return code


def dispatch_face(tok):
    return dispatch_wf("face_test.yml", tok)


def dispatch_wf(wf_file, tok):
    req = urllib.request.Request(
        f"https://api.github.com/repos/{REPO}/actions/workflows/{wf_file}/dispatches",
        headers={"Authorization": f"token {tok}",
                 "Accept": "application/vnd.github+json"},
        data=json.dumps({"ref": "main"}).encode(), method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.status


def dispatch_release(tok):
    req = urllib.request.Request(
        f"https://api.github.com/repos/{REPO}/actions/workflows/release.yml/dispatches",
        headers={"Authorization": f"token {tok}",
                 "Accept": "application/vnd.github+json"},
        data=json.dumps({"ref": "main"}).encode(), method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.status


def main():
    global ACCEPT_SHA
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-min", type=int, default=90)
    ap.add_argument("--no-release", action="store_true",
                    help="只验收不发包(发版须用户明示「发」,轨道默认仍会自动触发)")
    args = ap.parse_args()
    ACCEPT_SHA = subprocess.run(
        ["git", "rev-parse", "main"], capture_output=True,
        text=True, cwd=".").stdout.strip()
    tok = token()
    deadline = time.time() + args.max_min * 60
    rnd = 0
    line = "(尚无完整状态)"
    while time.time() < deadline:
        rnd += 1
        rids = latest_runs(tok, ACCEPT_SHA)
        for name, wf in MISSING_DISPATCH.items():
            if rids.get(name) is None:
                code = dispatch_wf(wf, tok)
                print(f"  {name} 缺席→补派发 ({code})", flush=True)
        states = {}
        for name, rid in rids.items():
            if rid:
                st = run_state(rid, tok)
                st["rid"] = rid
                states[name] = st
        if len(states) < len(WANT):
            missing = [n for n in WANT if n not in states]
            print(f"  等待触发: {missing}", flush=True)
            time.sleep(180)
            continue
        line = " | ".join(f"{s['name']}:{s['status']}/{s['conclusion']}/"
                          f"{s['steps']}步" for s in states.values())
        print(f"[第{rnd}轮] {line}", flush=True)

        reals = [s for s in states.values() if is_real_fail(s)]
        if reals:
            print("真失败(步骤已执行),停止轨道,不发包:\n" +
                  "\n".join(f"  {s['name']}" for s in reals), flush=True)
            sys.exit(2)
        if states and all(is_green(s) for s in states.values()):
            if args.no_release:
                print(f"{len(states)} 项全绿(--no-release:不自动发包,等用户口令)", flush=True)
                sys.exit(0)
            code = dispatch_release(tok)
            print(f"{len(states)} 项全绿 → 已触发 release.yml (HTTP {code})", flush=True)
            sys.exit(0)

        for name, st in states.items():
            if is_rejected(st):
                code = rerun(st["rid"], tok)
                print(f"  拒单→重派 {name} ({code})", flush=True)
        time.sleep(180)
    print(f"{args.max_min} 分钟未凑齐四绿,轨道放弃;当前:\n{line}", flush=True)
    sys.exit(3)


if __name__ == "__main__":
    main()
