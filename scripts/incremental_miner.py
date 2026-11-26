#!/usr/bin/env python3
"""Incremental PR miner with chronological environment bootstrapping.

Key features:
1. Candidate PRs are fetched and sorted chronologically (oldest -> newest).
2. Initial bootstrap container is seeded from the OLDEST PR's base_commit.
3. PRs are validated sequentially against the active bootstrap image.
4. If an environment crash occurs (validation fails due to broken runner/dependencies):
   - Diagnostic trace is logged loudly to stderr.
   - An incremental bootstrap is triggered at that PR's base_commit.
   - The new Docker image is committed, pushed to Docker Hub, and set as active.
   - The PR is re-tested with the new image.
5. On valid bug fixes (len(f2p) > 0):
   - Emits hard-mode ($out_dir/hard/<task>), easy-mode ($out_dir/easy/<task>), and fix-mode ($out_dir/fix/<task>)
     from the same single validation run!
6. Non-bugfix PRs (len(f2p) == 0 with clean test runs) are cleanly skipped without re-bootstrapping.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Ensure Repo2RLEnv/src is on PYTHONPATH
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "Repo2RLEnv" / "src"))

from repo2rlenv import github
from repo2rlenv.auth import resolve_repo_token
from repo2rlenv.bootstrap.cache import cache_key, load as load_bootstrap_cache
from repo2rlenv.bootstrap.docker import DockerSandbox
from repo2rlenv.bootstrap.runner import ensure_bootstrap
from repo2rlenv.bootstrap.spec import BootstrapResult
from repo2rlenv.emitter.harbor import HarborTask, write_harbor_task
from repo2rlenv.github import PullRequestSummary
from repo2rlenv.pipelines.pr_runtime_validate import ValidationOutcome, validate_pr
from repo2rlenv.pipelines.pr_swegemma_runtime import (
    _build_instruction,
    _diff_loc_changed,
    _difficulty_bucket,
    _files_in_patch,
    _linked_issue_number,
    _runtime_aux_files,
    build_eval_script,
    egress_guard_compose,
    fix_source_hosts,
    normalize_test_cmds_for_runtime,
    split_patch_and_test_patch,
    synthesize_task_instruction,
    targeted_test_cmds_for_pr,
)
from repo2rlenv.spec.input import AuthSpec, BootstrapSpec, LLMSpec, RepoSpec

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("incremental_miner")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Incremental PR miner with chronological bootstrapping")
    parser.add_argument("--repo", required=True, help="GitHub repo (e.g. tiangolo/fastapi)")
    parser.add_argument("--out", default="train", help="Base output directory")
    parser.add_argument("--limit", type=int, default=100, help="Max PRs to fetch from GitHub")
    parser.add_argument("--since", default="2024-01-01", help="Only mine PRs merged on or after this date (YYYY-MM-DD)")
    parser.add_argument("--reverse", action="store_true", help="Reverse PR ordering (process newest first)")
    parser.add_argument("--max-success", type=int, default=0, help="Stop after emitting this many verified tasks (0 = no limit)")
    parser.add_argument("--prs", default="", help="Comma-separated list of PR numbers to process (e.g. 11194,11355)")
    parser.add_argument("--exclude-prs", default="", help="Comma-separated list of PR numbers to exclude (e.g. 3672,11194)")
    parser.add_argument(
        "--exclude-dirs",
        default="test,data/test",
        help="Comma-separated task directories to scan and exclude test PRs from",
    )
    parser.add_argument("--base-image", default="", help="Optional base Docker image to override initial bootstrap")
    parser.add_argument("--max-source-files", type=int, default=30, help="Max source files in PR diff")
    parser.add_argument("--llm", default="openai/thinkingmachines/Inkling-Small", help="LLM model string")
    parser.add_argument(
        "--llm-endpoint",
        default="https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1",
        help="LLM API endpoint",
    )
    parser.add_argument("--llm-key-env", default="TINKER_API_KEY", help="Env var name for API key")
    parser.add_argument("--reasoning-effort", default="high", help="LLM reasoning effort")
    parser.add_argument("--image-registry", default="docker.io/carlesoctav", help="Docker registry prefix")
    parser.add_argument("--cache-dir", default="workspace/bootstrap", help="Cache dir for bootstrap specs")
    parser.add_argument("--max-iterations", type=int, default=100000, help="Bootstrap max agent iterations")
    parser.add_argument("--max-seconds", type=int, default=86400, help="Bootstrap max seconds")
    parser.add_argument("--max-spend-usd", type=float, default=0.0, help="Bootstrap max USD spend")
    return parser.parse_args()


def is_environment_crash(outcome: ValidationOutcome) -> bool:
    """Check if validation failed due to an environment / runner crash."""
    if outcome.status == "verified":
        return False
    if outcome.reason == "no fail-to-pass tests after validation":
        return False
    # If the patch simply didn't apply to git, it's not a container environment crash
    if "failed to apply at base_commit" in (outcome.reason or ""):
        return False
    # If 0 tests were ever collected or executed in post-fix
    if len(outcome.fail_to_pass) == 0 and len(outcome.pass_to_pass) == 0:
        return True
    # Look for fatal unhandled tracebacks in logs
    post = (outcome.post_log or "").lower()
    pre = (outcome.pre_log or "").lower()
    for marker in ("traceback (most recent call last)", "typeerror:", "importerror:", "modulenotfounderror:"):
        if marker in post or marker in pre:
            return True
    return False


def start_sandbox_for_image(image_tag: str, repo_url: str, language: str = "python") -> DockerSandbox:
    """Start a Docker sandbox and ensure /workspace is initialized."""
    logger.info("Starting Docker sandbox from image: %s", image_tag)
    sandbox = DockerSandbox.start(base_image=image_tag, repo_dir=None)
    sandbox.exec("git config --global --add safe.directory /workspace", timeout=10)
    check_git = sandbox.exec("test -d /workspace/.git && echo OK", timeout=10)
    if not (check_git.ok and "OK" in check_git.stdout):
        logger.info("Cloning %s into sandbox /workspace...", repo_url)
        clone_res = sandbox.exec(
            f"rm -rf /workspace/.keep && git clone {repo_url} /workspace",
            timeout=300,
        )
        if not clone_res.ok:
            raise RuntimeError(f"Failed to clone repository inside container: {clone_res.stderr}")
        if language == "python":
            sandbox.exec(
                "cd /workspace && pip install -e . --no-build-isolation --find-links /wheels 2>/dev/null || true",
                timeout=180,
            )
    return sandbox


def emit_tasks_all_modes(
    *,
    pr: PullRequestSummary,
    patch: str,
    test_patch: str,
    fail_to_pass: list[str],
    pass_to_pass: list[str],
    bootstrap: BootstrapResult,
    repo_spec: RepoSpec,
    llm_spec: LLMSpec,
    out_dir: Path,
    token: str | None,
    pre_log: str = "",
) -> None:
    """Emit hard, easy, and fix-desc task variants from a single verified validation."""
    owner, name = repo_spec.owner_name
    task_id = f"{owner}__{name}-{pr.number}"

    resolved_test_cmds = targeted_test_cmds_for_pr(
        normalize_test_cmds_for_runtime(bootstrap.test_cmds),
        _files_in_patch(test_patch),
    )
    eval_script = build_eval_script(
        base_commit=pr.base_sha,
        test_patch=test_patch,
        test_cmds=resolved_test_cmds,
        language=bootstrap.language.value,
        fail_to_pass=fail_to_pass,
        pass_to_pass=pass_to_pass,
    )
    image_ref = bootstrap.image_digest if bootstrap.pushed_to_registry else bootstrap.image_tag

    runtime_metadata = {
        "pr_url": pr.url,
        "pr_merged_at": pr.merged_at,
        "base_commit": pr.base_sha,
        "fail_to_pass": fail_to_pass,
        "pass_to_pass": pass_to_pass,
        "validation_status": "verified",
        "bootstrap_image": bootstrap.image_digest,
        "reward_mode": "graded",
    }
    repo2env = {
        "pipeline": "pr_swegemma_runtime",
        "pipeline_version": "0.3.0",
        "repo": f"{owner}/{name}",
        "ref": pr.base_sha,
        "reference": pr.url,
        "source_access": repo_spec.access,
        "built_at": datetime.now(UTC).isoformat(),
        "synthesis_llm": llm_spec.qualified_name if llm_spec else "none",
        "reward_kinds": ["test_execution", "diff_similarity"],
        "pr_swegemma_runtime": runtime_metadata,
        "pr_runtime": runtime_metadata,
        "reward_calibration": {
            "f2p_count": len(fail_to_pass),
            "p2p_count": len(pass_to_pass),
            "source_files": len(_files_in_patch(patch)),
            "loc_changed": _diff_loc_changed(patch),
            "difficulty": _difficulty_bucket(len(fail_to_pass), _diff_loc_changed(patch)),
        },
    }
    aux_files = {
        **_runtime_aux_files(fail_to_pass, pass_to_pass),
        "environment/docker-compose.yaml": egress_guard_compose(
            fix_source_hosts(repo_spec.source_kind),
            base_commit=pr.base_sha,
        ),
    }

    env_dockerfile = bootstrap.dockerfile_reconstruction if bootstrap.dockerfile_reconstruction else None

    # 1. Hard mode: PR title + body
    instr_hard = _build_instruction(pr, owner, name, token=token, hard_mode=True)
    task_hard = HarborTask(
        name=task_id,
        org="default",
        description=pr.title or task_id,
        instruction=instr_hard,
        oracle_diff=patch,
        repo2env=repo2env,
        difficulty="medium",
        category="bugfix",
        keywords=[name, "pr_swegemma_runtime", "hard"],
        docker_image=image_ref,
        environment_dockerfile=env_dockerfile,
        test_script=eval_script,
        aux_files=aux_files,
    )
    hard_path = write_harbor_task(task_hard, out_dir / "hard")

    # 2. Easy mode: linked issue title + body (with fallback to PR body)
    instr_easy = _build_instruction(pr, owner, name, token=token, hard_mode=False)
    task_easy = HarborTask(
        name=task_id,
        org="default",
        description=pr.title or task_id,
        instruction=instr_easy,
        oracle_diff=patch,
        repo2env=repo2env,
        difficulty="medium",
        category="bugfix",
        keywords=[name, "pr_swegemma_runtime", "easy"],
        docker_image=image_ref,
        environment_dockerfile=env_dockerfile,
        test_script=eval_script,
        aux_files=aux_files,
    )
    easy_path = write_harbor_task(task_easy, out_dir / "easy")

    # 3. Fix-desc mode: LLM-synthesized clean bug report grounded in pre_log + fail_to_pass
    issue_title = ""
    issue_body = ""
    issue_num = _linked_issue_number(pr.body or "")
    if issue_num is not None and owner and name:
        fetched = github.fetch_issue(owner, name, issue_num, token=token)
        if fetched:
            issue_title, issue_body = fetched

    instr_fix = synthesize_task_instruction(
        pr_title=pr.title,
        pr_body=pr.body or "",
        base_commit=pr.base_sha,
        fail_to_pass=fail_to_pass,
        pre_log=pre_log,
        issue_title=issue_title,
        issue_body=issue_body,
        patch=patch,
        test_patch=test_patch,
        llm_spec=llm_spec,
    )
    task_fix_desc = HarborTask(
        name=task_id,
        org="default",
        description=pr.title or task_id,
        instruction=instr_fix,
        oracle_diff=patch,
        repo2env=repo2env,
        difficulty="medium",
        category="bugfix",
        keywords=[name, "pr_swegemma_runtime", "fix-desc"],
        docker_image=image_ref,
        environment_dockerfile=env_dockerfile,
        test_script=eval_script,
        aux_files=aux_files,
    )
    fix_desc_path = write_harbor_task(task_fix_desc, out_dir / "fix-desc")

    logger.info("Emitted %s -> hard (%s), easy (%s), and fix-desc (%s)", task_id, hard_path, easy_path, fix_desc_path)


emit_tasks_both_modes = emit_tasks_all_modes


def main():
    args = parse_args()
    owner, _, name = args.repo.partition("/")
    repo_url = f"https://github.com/{args.repo}"
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    auth = AuthSpec()
    repo_spec = RepoSpec(url=repo_url, ref="HEAD")
    token = resolve_repo_token(repo_spec, auth)

    provider, _, model = args.llm.partition("/")
    llm_spec = LLMSpec(
        provider=provider,
        model=model,
        endpoint=args.llm_endpoint,
        api_key_env=args.llm_key_env,
        reasoning_effort=args.reasoning_effort,
        timeout_sec=600,
    )

    spend_cap = args.max_spend_usd if args.max_spend_usd and args.max_spend_usd > 0 else None
    bspec = BootstrapSpec(
        enabled=True,
        image_registry=args.image_registry,
        cache_dir=Path(args.cache_dir).resolve(),
        max_iterations=args.max_iterations,
        max_seconds=args.max_seconds,
        max_llm_spend_usd=spend_cap,
    )

    raw_prs: list[PullRequestSummary] = []
    if args.prs.strip():
        pr_numbers = [int(p.strip()) for p in args.prs.split(",") if p.strip()]
        logger.info("Fetching %d specified PRs for %s: %s", len(pr_numbers), args.repo, pr_numbers[:10])
        for num in pr_numbers:
            summary = github.fetch_pr_summary(owner, name, num, token=token)
            if summary is not None:
                raw_prs.append(summary)
            else:
                logger.warning("Failed to fetch summary for PR #%d", num)
    else:
        logger.info("Fetching merged PRs for %s (limit=%d, since=%s)...", args.repo, args.limit, args.since or "all")
        since_date = None
        if args.since.strip():
            since_date = datetime.strptime(args.since.strip(), "%Y-%m-%d").date()
        raw_prs = github.list_merged_prs(owner, name, limit=args.limit, since=since_date, token=token)

    if not raw_prs:
        raise SystemExit(f"No merged PRs found for {args.repo}")

    # Collect PRs to exclude (e.g. from test suites to avoid test contamination)
    exclude_prs: set[int] = set()
    if args.exclude_prs.strip():
        for p in args.exclude_prs.split(","):
            p = p.strip()
            if p.isdigit():
                exclude_prs.add(int(p))
    if args.exclude_dirs.strip():
        for d_str in args.exclude_dirs.split(","):
            d_path = Path(d_str.strip())
            if not d_path.is_absolute():
                d_path = REPO_ROOT / d_path
            if not d_path.exists():
                continue
            for task_dir in d_path.rglob("*"):
                if not task_dir.is_dir():
                    continue
                dir_name = task_dir.name
                m = re.search(r"-(\d+)$", dir_name)
                if m:
                    pr_num = int(m.group(1))
                    prefix = dir_name[: m.start()].lower()
                    if name.lower() in prefix or (owner and owner.lower() in prefix):
                        exclude_prs.add(pr_num)
    if exclude_prs:
        logger.info(
            "Found %d test PRs to exclude for %s: %s",
            len(exclude_prs),
            args.repo,
            sorted(list(exclude_prs))[:15],
        )

    # Sort PRs: OLDEST -> NEWEST by default, or NEWEST -> OLDEST if --reverse
    prs = sorted(
        [
            p
            for p in raw_prs
            if p.merged_at
            and not p.is_draft
            and p.number not in exclude_prs
            and (not args.since.strip() or p.merged_at[:10] >= args.since.strip())
        ],
        key=lambda p: p.merged_at,
        reverse=args.reverse,
    )
    if exclude_prs:
        logger.info(
            "Candidate PRs after excluding test PRs: %d (excluded %d)",
            len(prs),
            len(raw_prs) - len(prs),
        )
    order_label = "newest to oldest" if args.reverse else "oldest to newest"
    logger.info(
        "Collected %d merged PRs (%s). Date range: %s to %s",
        len(prs),
        order_label,
        prs[0].merged_at,
        prs[-1].merged_at,
    )

    # Seed bootstrap image with the initial PR commit
    initial_pr = prs[0]
    if args.base_image.strip():
        bspec.base_image = args.base_image.strip()

    logger.info(
        "Seeding initial bootstrap on initial PR #%d (base_commit: %s, date: %s)...",
        initial_pr.number,
        initial_pr.base_sha[:12],
        initial_pr.merged_at,
    )
    seed_repo_spec = RepoSpec(url=repo_url, ref=initial_pr.base_sha)
    current_bootstrap = ensure_bootstrap(seed_repo_spec, bspec, llm_spec, auth)
    logger.info("Active bootstrap image: %s", current_bootstrap.image_tag)

    sandbox = start_sandbox_for_image(current_bootstrap.image_tag, repo_url, language=current_bootstrap.language.value)

    emitted_count = 0
    skipped_count = 0
    rebootstrapped_count = 0

    try:
        for idx, pr in enumerate(prs, 1):
            logger.info(
                "[%d/%d] Inspecting PR #%d (%s, base: %s)...",
                idx,
                len(prs),
                pr.number,
                pr.merged_at,
                pr.base_sha[:12],
            )

            task_id = f"{owner}__{name}-{pr.number}"
            if (out_dir / "hard" / task_id).exists():
                logger.info("PR #%d: already verified & emitted (%s), skipping", pr.number, task_id)
                emitted_count += 1
                if args.max_success > 0 and emitted_count >= args.max_success:
                    logger.info("Reached target success limit (%d verified tasks). Stopping.", args.max_success)
                    break
                continue

            try:
                diff = github.fetch_pr_diff(owner, name, pr.number, token=token)
            except github.GitHubError as exc:
                logger.warning("PR #%d: diff fetch failed: %s", pr.number, exc)
                skipped_count += 1
                continue
            patch, test_patch = split_patch_and_test_patch(diff)
            if not patch.strip() or not test_patch.strip():
                logger.info("PR #%d: skipped (empty patch or no test patch)", pr.number)
                skipped_count += 1
                continue

            changed_files = _files_in_patch(patch)
            if len(changed_files) > args.max_source_files:
                logger.info("PR #%d: skipped (touches %d source files > %d)", pr.number, len(changed_files), args.max_source_files)
                skipped_count += 1
                continue

            targeted_cmds = targeted_test_cmds_for_pr(
                normalize_test_cmds_for_runtime(current_bootstrap.test_cmds),
                _files_in_patch(test_patch),
            )

            outcome = validate_pr(
                sandbox=sandbox,
                base_commit=pr.base_sha,
                patch=patch,
                test_patch=test_patch,
                test_cmds=targeted_cmds,
                language=current_bootstrap.language.value,
            )

            if outcome.reason == "no fail-to-pass tests after validation":
                logger.info(
                    "PR #%d: skipped (status=%s, f2p=%d, p2p=%d, reason=%s)",
                    pr.number,
                    outcome.status,
                    len(outcome.fail_to_pass),
                    len(outcome.pass_to_pass),
                    outcome.reason,
                )
                skipped_count += 1
                continue

            # Check for environment crash
            if is_environment_crash(outcome):
                logger.warning("=" * 70)
                logger.warning(
                    "ENVIRONMENT BREAK DETECTED on PR #%d at commit %s (%s)!",
                    pr.number,
                    pr.base_sha[:12],
                    pr.merged_at,
                )
                logger.warning("Validation status: %s, Reason: %s", outcome.status, outcome.reason)
                excerpt = (outcome.post_log or outcome.pre_log or "").strip()
                if excerpt:
                    last_lines = "\n".join(excerpt.splitlines()[-20:])
                    logger.warning("Traceback excerpt:\n%s", last_lines)
                logger.warning("=" * 70)

                # Re-bootstrap image incrementally at this commit
                logger.info("Triggering incremental bootstrap at commit %s...", pr.base_sha[:12])
                new_repo_spec = RepoSpec(url=repo_url, ref=pr.base_sha)
                sandbox.cleanup()

                current_bootstrap = ensure_bootstrap(
                    new_repo_spec,
                    bspec,
                    llm_spec,
                    auth,
                    force=True,
                )
                rebootstrapped_count += 1
                logger.info("Promoting new bootstrap image to active: %s", current_bootstrap.image_tag)

                # Restart sandbox with new image
                sandbox = start_sandbox_for_image(current_bootstrap.image_tag, repo_url, language=current_bootstrap.language.value)

                # Retry validation with new image
                logger.info("Retrying PR #%d with new bootstrap image...", pr.number)
                targeted_cmds = targeted_test_cmds_for_pr(
                    normalize_test_cmds_for_runtime(current_bootstrap.test_cmds),
                    _files_in_patch(test_patch),
                )
                outcome = validate_pr(
                    sandbox=sandbox,
                    base_commit=pr.base_sha,
                    patch=patch,
                    test_patch=test_patch,
                    test_cmds=targeted_cmds,
                    language=current_bootstrap.language.value,
                )

            # Evaluate outcome after potential re-bootstrap
            if outcome.status == "verified" and len(outcome.fail_to_pass) > 0:
                emit_tasks_all_modes(
                    pr=pr,
                    patch=patch,
                    test_patch=test_patch,
                    fail_to_pass=outcome.fail_to_pass,
                    pass_to_pass=outcome.pass_to_pass,
                    bootstrap=current_bootstrap,
                    repo_spec=repo_spec,
                    llm_spec=llm_spec,
                    out_dir=out_dir,
                    token=token,
                    pre_log=outcome.pre_log,
                )
                emitted_count += 1
                if args.max_success > 0 and emitted_count >= args.max_success:
                    logger.info(
                        "Reached target success limit (%d verified tasks) for %s. Stopping repo mining.",
                        args.max_success,
                        args.repo,
                    )
                    break
            else:
                logger.info(
                    "PR #%d: skipped (status=%s, f2p=%d, p2p=%d, reason=%s)",
                    pr.number,
                    outcome.status,
                    len(outcome.fail_to_pass),
                    len(outcome.pass_to_pass),
                    outcome.reason,
                )
                skipped_count += 1

    finally:
        logger.info("Cleaning up active Docker sandbox...")
        sandbox.cleanup()

    logger.info("=" * 60)
    logger.info("Incremental mining complete for %s!", args.repo)
    logger.info("  Total PRs processed: %d", len(prs))
    logger.info("  Verified & Emitted:  %d (hard, easy, and fix)", emitted_count)
    logger.info("  Skipped:             %d", skipped_count)
    logger.info("  Re-bootstrapped:     %d times", rebootstrapped_count)
    logger.info("  Output directories:  %s/hard, %s/easy, and %s/fix", out_dir, out_dir, out_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
