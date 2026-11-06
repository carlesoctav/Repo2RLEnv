#!/usr/bin/env python3
"""Batch instruction synthesizer for Harbor tasks (converts easy/hard to fix mode)."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import logging
import re
import shutil
import sys
import time
import tomllib
from pathlib import Path

import tomli_w

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "Repo2RLEnv" / "src"))

from repo2rlenv import github
from repo2rlenv.auth import resolve_repo_token
from repo2rlenv.pipelines.pr_swegemma_runtime import (
    _linked_issue_number,
    synthesize_task_instruction,
)
from repo2rlenv.spec.input import AuthSpec, LLMSpec, RepoSpec

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("synthesize_instructions")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthesize clean bug reports for Harbor tasks")
    parser.add_argument("--in-dir", default="", help="Input task directory or parent directory containing tasks")
    parser.add_argument("--out-dir", default="", help="Output directory for synthesized tasks (e.g. test/fix-desc)")
    parser.add_argument("--tasks-dir", default="", help="Base tasks directory containing easy/ or hard/ subdirectories")
    parser.add_argument("--tasks", default="", help="Comma-separated task names to filter (e.g. tiangolo__fastapi-11194)")
    parser.add_argument("--llm", default="openai/thinkingmachines/Inkling-Small", help="LLM model string")
    parser.add_argument(
        "--llm-endpoint",
        default="https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1",
        help="LLM API endpoint",
    )
    parser.add_argument("--llm-key-env", default="TINKER_API_KEY", help="Env var name for API key")
    parser.add_argument("--reasoning-effort", default="high", help="LLM reasoning effort")
    parser.add_argument("--concurrency", type=int, default=8, help="Max concurrent synthesis workers")
    parser.add_argument("--version", default="old", choices=["old", "v2"], help="SWE synthesis prompt recipe version (old vs v2)")
    parser.add_argument("--force", action="store_true", help="Force regenerate even if task already exists in out-dir")
    return parser.parse_args()


def resolve_dirs(args: argparse.Namespace) -> tuple[Path, Path | None]:
    if args.tasks_dir:
        base = Path(args.tasks_dir)
        if (base / "easy").is_dir():
            in_dir = base / "easy"
        elif (base / "hard").is_dir():
            in_dir = base / "hard"
        else:
            in_dir = base
        out_dir = Path(args.out_dir) if args.out_dir else base / "fix-desc"
        return in_dir, out_dir

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir) if args.out_dir else None
    return in_dir, out_dir


def find_task_dirs(in_dir: Path) -> list[Path]:
    if (in_dir / "task.toml").is_file() and (in_dir / "instruction.md").is_file():
        return [in_dir]
    task_dirs = [
        d for d in in_dir.iterdir()
        if d.is_dir() and (d / "task.toml").is_file() and (d / "instruction.md").is_file()
    ]
    return sorted(task_dirs, key=lambda p: p.name)


def process_task(
    task_path: Path,
    dest_dir: Path,
    llm_spec: LLMSpec,
    auth: AuthSpec,
    version: str = "old",
) -> None:
    with open(task_path / "task.toml", "rb") as f:
        toml_data = tomllib.load(f)

    repo = toml_data["metadata"]["repo2env"]["repo"]
    base_commit = toml_data["metadata"]["repo2env"]["ref"]
    fail_to_pass = toml_data["metadata"]["repo2env"]["pr_swegemma_runtime"]["fail_to_pass"]
    oracle_diff = (task_path / "solution" / "patch.diff").read_text(encoding="utf-8")
    instruction = (task_path / "instruction.md").read_text(encoding="utf-8")

    pr_title = toml_data["task"]["description"]
    m_title = re.search(r"\*\*Title:\*\*\s*(.*)", instruction)
    if m_title:
        pr_title = m_title.group(1).strip()
    m_desc = re.search(r"## Description\s*\n\n(.*?)\n\n## Task", instruction, re.DOTALL)
    pr_body = m_desc.group(1).strip() if m_desc else ""

    repo_spec = RepoSpec(url=f"https://github.com/{repo}", ref=base_commit)
    token = resolve_repo_token(repo_spec, auth)
    owner, name = repo.split("/")

    issue_title = ""
    issue_body = ""
    issue_num = _linked_issue_number(pr_body)
    if issue_num is not None and token:
        fetched = github.fetch_issue(owner, name, issue_num, token=token)
        if fetched:
            issue_title, issue_body = fetched

    test_sh_path = task_path / "tests" / "test.sh"
    test_patch = ""
    if test_sh_path.is_file():
        m_tp = re.search(
            r"<<'EOF_R2E_TEST_PATCH'\n(.*?)\nEOF_R2E_TEST_PATCH",
            test_sh_path.read_text(encoding="utf-8"),
            re.DOTALL,
        )
        if m_tp:
            test_patch = m_tp.group(1).strip()

    logger.info("Synthesizing instruction for %s...", task_path.name)
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            new_instruction = synthesize_task_instruction(
                pr_title=pr_title,
                pr_body=pr_body,
                base_commit=base_commit,
                fail_to_pass=fail_to_pass,
                pre_log="",
                issue_title=issue_title,
                issue_body=issue_body,
                patch=oracle_diff,
                test_patch=test_patch,
                llm_spec=llm_spec,
                version=version,
            )
            break
        except Exception as e:
            if attempt == max_retries:
                raise
            logger.warning("Synthesis call failed for %s (attempt %d/%d): %s. Retrying in 4s...", task_path.name, attempt, max_retries, e)
            time.sleep(4)

    if dest_dir.resolve() != task_path.resolve():
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(task_path, dest_dir, dirs_exist_ok=True)

    (dest_dir / "instruction.md").write_text(new_instruction, encoding="utf-8")

    h = hashlib.sha256()
    h.update(new_instruction.encode("utf-8"))
    h.update(b"\0")
    h.update(oracle_diff.encode("utf-8"))
    new_hash = f"sha256:{h.hexdigest()}"

    toml_data["metadata"]["repo2env"]["content_hash"] = new_hash
    toml_data["metadata"]["repo2env"]["synthesis_llm"] = llm_spec.qualified_name

    keywords = [k for k in toml_data["metadata"]["keywords"] if k not in ("easy", "hard", "fix", "fix-description")]
    if "fix-desc" not in keywords:
        keywords.append("fix-desc")
    toml_data["metadata"]["keywords"] = keywords

    (dest_dir / "task.toml").write_bytes(tomli_w.dumps(toml_data).encode("utf-8"))
    logger.info("Wrote %s (hash=%s)", dest_dir, new_hash[:19])


def main() -> None:
    args = parse_args()
    in_dir, out_dir = resolve_dirs(args)
    task_dirs = find_task_dirs(in_dir)

    if args.tasks:
        filter_names = {t.strip() for t in args.tasks.split(",") if t.strip()}
        task_dirs = [d for d in task_dirs if d.name in filter_names]

    if not task_dirs:
        logger.warning("No task directories found in %s", in_dir)
        return

    provider, _, model = args.llm.partition("/")
    llm_spec = LLMSpec(
        provider=provider,
        model=model,
        endpoint=args.llm_endpoint,
        api_key_env=args.llm_key_env,
        reasoning_effort=args.reasoning_effort,
    )
    auth = AuthSpec()

    tasks_to_run = []
    for td in task_dirs:
        dest_td = out_dir / td.name if out_dir and (len(task_dirs) > 1 or out_dir.name != td.name) else (out_dir if out_dir else td)
        if not args.force and (dest_td / "task.toml").is_file() and (dest_td / "instruction.md").is_file():
            continue
        tasks_to_run.append((td, dest_td))

    logger.info(
        "Synthesizing instructions for %d/%d task(s) from %s (concurrency=%d)...",
        len(tasks_to_run),
        len(task_dirs),
        in_dir,
        args.concurrency,
    )

    if not tasks_to_run:
        logger.info("All tasks already synthesized in %s.", out_dir or in_dir)
        return

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(process_task, td, dest_td, llm_spec, auth, version=args.version): (td.name, dest_td)
            for td, dest_td in tasks_to_run
        }
        completed_count = 0
        for fut in concurrent.futures.as_completed(futures):
            t_name, d_td = futures[fut]
            completed_count += 1
            try:
                fut.result()
                logger.info("[%d/%d] Successfully synthesized %s", completed_count, len(tasks_to_run), t_name)
            except Exception as e:
                logger.error("[%d/%d] Failed synthesizing %s: %s", completed_count, len(tasks_to_run), t_name, e)

    logger.info("All %d tasks processed successfully.", len(tasks_to_run))


if __name__ == "__main__":
    main()
