#!/usr/bin/env python3
"""Convert TIGER-Lab/SWE-Next benchmark into Harbor-compliant tasks.

Emits self-contained Harbor tasks with:
  train/<instance_id>/
  ├── task.toml
  ├── instruction.md
  ├── environment/
  │   └── docker-compose.yaml
  ├── solution/
  │   ├── patch.diff
  │   └── solve.sh
  └── tests/
      ├── f2p.json
      ├── p2p.json
      ├── test.sh
      └── verifier.py
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
import sys
from typing import Any

from datasets import load_dataset
import tomli_w

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "Repo2RLEnv" / "src"))


def get_verifier_source() -> str:
    verifier_path = REPO_ROOT / "Repo2RLEnv" / "src" / "repo2rlenv" / "pipelines" / "_pr_runtime_verifier.py"
    return verifier_path.read_text(encoding="utf-8")


def extract_description(problem_statement: str) -> str:
    lines = [line.strip() for line in problem_statement.splitlines() if line.strip()]
    first_line = lines[0] if lines else "Bugfix task"
    if first_line.lower().startswith("title:"):
        return first_line.split(":", 1)[1].strip()
    return first_line[:120]


def resolve_full_test_paths(row: dict[str, Any]) -> tuple[list[str], list[str]]:
    erc = json.loads(row["execution_result_content"])
    log = (erc.get("old_commit_res_stdout") or "") + "\n" + (erc.get("new_commit_res_stdout") or "")
    from repo2rlenv.pipelines._pr_runtime_verifier import parse_pytest
    smap = parse_pytest(log)
    
    test_files = erc.get("test_file_names") or ["test_1.py"]
    default_prefix = f"r2e_tests/{test_files[0]}::"

    def _resolve(t: str) -> str:
        if t in smap:
            return t
        for k in smap:
            if k.endswith("::" + t) or k.split("::", 1)[-1] == t:
                return k
        return f"{default_prefix}{t}"

    f2p = [_resolve(t) for t in row["FAIL_TO_PASS"]]
    p2p = [_resolve(t) for t in (row["PASS_TO_PASS"] or [])]
    return f2p, p2p


def build_task_toml(row: dict[str, Any], docker_image: str, f2p: list[str], p2p: list[str]) -> str:
    instance_id = row["instance_id"]
    repo = row["repo"]
    description = extract_description(row["problem_statement"])
    keywords = [row.get("repo_key") or repo.split("/")[-1], "swe-next", "bugfix"]

    payload = {
        "version": "1.0",
        "task": {
            "name": f"default/{instance_id}",
            "description": description,
        },
        "metadata": {
            "difficulty": row.get("difficulty") or "medium",
            "category": "bugfix",
            "keywords": keywords,
            "repo2env": {
                "pipeline": "swe_next",
                "repo": repo,
                "base_commit": row["base_commit"],
                "fail_to_pass": f2p,
                "pass_to_pass": p2p,
                "reward_kinds": ["test_execution", "diff_similarity"],
            },
        },
        "agent": {
            "timeout_sec": 1800.0,
        },
        "verifier": {
            "timeout_sec": 300.0,
        },
        "environment": {
            "docker_image": docker_image,
        },
    }
    return tomli_w.dumps(payload)


def build_instruction(problem_statement: str) -> str:
    return (
        f"{problem_statement.strip()}\n\n"
        "## Task\n\n"
        "Modify the repository so that the issue described above is resolved. "
        "The task's test suite verifies your patch by running the test suite.\n"
    )


def build_docker_compose(docker_image: str) -> str:
    return f"""# Anti-contamination egress guard and workspace setup for SWE-Next
services:
  main:
    image: "{docker_image}"
    extra_hosts:
      - "pypi.org:0.0.0.0"
      - "pypi.python.org:0.0.0.0"
      - "files.pythonhosted.org:0.0.0.0"
      - "github.com:0.0.0.0"
      - "api.github.com:0.0.0.0"
      - "raw.githubusercontent.com:0.0.0.0"
      - "codeload.github.com:0.0.0.0"
      - "objects.githubusercontent.com:0.0.0.0"
    entrypoint:
      - "bash"
      - "-c"
      - |
        set -e
        git config --global --add safe.directory /workspace
        git config --global user.email "agent@eval.local"
        git config --global user.name "Agent"
        git config --global init.defaultBranch main
        cd /workspace
        if [ ! -d /workspace/.git ]; then
            git init
            rm -f /workspace/execution_result.json /workspace/expected_test_output.json /workspace/parsed_commit.json /workspace/syn_issue.json /workspace/modified_*.json /workspace/install.sh
            git add -A
            git commit -m "initial baseline" --allow-empty
            git checkout -q -B base
        fi
        touch /ready
        if [ $$# -eq 0 ]; then exec sleep infinity; else exec "$$@"; fi
      - "bash"
"""


def build_solve_sh() -> str:
    return """#!/bin/bash
set -euxo pipefail
for i in $(seq 1 120); do
  if [ -f /ready ]; then break; fi
  sleep 0.5
done
cd /workspace
git config --global --add safe.directory /workspace
PATCH="$(dirname "$0")/patch.diff"
if ! git apply --verbose --reject "$PATCH" 2>/dev/null; then
  patch -p1 --batch --forward -i "$PATCH"
fi
"""


def build_test_sh() -> str:
    return """#!/bin/bash
set -uxo pipefail
for i in $(seq 1 120); do
  if [ -f /ready ]; then break; fi
  sleep 0.5
done
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd /workspace
git config --global --add safe.directory /workspace
mkdir -p /logs/verifier
( bash /workspace/run_tests.sh ) > /logs/verifier/test_output.log 2>&1
TEST_EXIT_CODE=$?
cat /logs/verifier/test_output.log
python3 "$SCRIPT_DIR/verifier.py" --log /logs/verifier/test_output.log --f2p "$SCRIPT_DIR/f2p.json" --p2p "$SCRIPT_DIR/p2p.json" --test-cmds 'bash /workspace/run_tests.sh' --runner pytest --exit-code "$TEST_EXIT_CODE" --out-dir /logs/verifier || { [ "$TEST_EXIT_CODE" -eq 0 ] && echo "1.0" > /logs/verifier/reward.txt || echo "0.0" > /logs/verifier/reward.txt; }
exit 0
"""


def convert_task(row: dict[str, Any], dest_dir: Path, verifier_content: str, overwrite: bool = False) -> str:
    instance_id = row["instance_id"]
    task_dir = dest_dir / instance_id
    if task_dir.exists() and not overwrite:
        return instance_id

    task_dir.mkdir(parents=True, exist_ok=True)
    env_dir = task_dir / "environment"
    env_dir.mkdir(exist_ok=True)
    sol_dir = task_dir / "solution"
    sol_dir.mkdir(exist_ok=True)
    tests_dir = task_dir / "tests"
    tests_dir.mkdir(exist_ok=True)

    raw_image = row["docker_image"]
    docker_image = f"docker.io/{raw_image}" if not raw_image.startswith("docker.io/") else raw_image

    f2p_full, p2p_full = resolve_full_test_paths(row)

    (task_dir / "task.toml").write_text(build_task_toml(row, docker_image, f2p_full, p2p_full), encoding="utf-8")
    (task_dir / "instruction.md").write_text(build_instruction(row["problem_statement"]), encoding="utf-8")
    (env_dir / "docker-compose.yaml").write_text(build_docker_compose(docker_image), encoding="utf-8")

    (sol_dir / "patch.diff").write_text(row["patch"], encoding="utf-8")
    solve_path = sol_dir / "solve.sh"
    solve_path.write_text(build_solve_sh(), encoding="utf-8")
    solve_path.chmod(0o755)

    (tests_dir / "f2p.json").write_text(json.dumps(f2p_full, indent=2), encoding="utf-8")
    (tests_dir / "p2p.json").write_text(json.dumps(p2p_full, indent=2), encoding="utf-8")
    (tests_dir / "verifier.py").write_text(verifier_content, encoding="utf-8")
    test_path = tests_dir / "test.sh"
    test_path.write_text(build_test_sh(), encoding="utf-8")
    test_path.chmod(0o755)

    return instance_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert TIGER-Lab/SWE-Next to Harbor benchmark tasks")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "train", help="Output directory (default: train)")
    parser.add_argument("--dataset", type=str, default="TIGER-Lab/SWE-Next", help="HuggingFace dataset path")
    parser.add_argument("--split", type=str, default="train", help="Dataset split (default: train)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tasks to convert")
    parser.add_argument("--instance-ids", nargs="*", default=None, help="Specific instance IDs to convert")
    parser.add_argument("--workers", type=int, default=16, help="Parallel worker threads")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing task folders")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    verifier_content = get_verifier_source()

    print(f"Loading dataset {args.dataset} (split: {args.split})...")
    ds = load_dataset(args.dataset, split=args.split)
    print(f"Loaded {len(ds)} items.")

    if args.instance_ids:
        target_ids = set(args.instance_ids)
        tasks = [row for row in ds if row["instance_id"] in target_ids]
    elif args.limit:
        tasks = [ds[i] for i in range(min(args.limit, len(ds)))]
    else:
        tasks = [row for row in ds]

    print(f"Converting {len(tasks)} tasks to {args.out_dir} with {args.workers} workers...")
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(convert_task, row, args.out_dir, verifier_content, args.overwrite)
            for row in tasks
        ]
        for f in as_completed(futures):
            f.result()
            completed += 1
            if completed % 100 == 0 or completed == len(tasks):
                print(f"[{completed}/{len(tasks)}] Converted tasks...")

    print(f"Successfully converted {completed} tasks into {args.out_dir}.")


if __name__ == "__main__":
    main()
