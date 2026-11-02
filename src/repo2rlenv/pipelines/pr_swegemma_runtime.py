"""Sandbox-verified PR mining (SWE-bench-style).

For each merged PR within scope:
  1. Pull metadata + unified diff via `gh pr list` / `gh pr diff`
  2. Split the diff into `patch` (source files) and `test_patch` (test files)
     using the same keyword-on-path heuristic SWE-bench uses
  3. If validation enabled: run the bootstrap container twice (once with
     test_patch only, once with both patches) to compute FAIL_TO_PASS and
     PASS_TO_PASS sets — the verified oracle
  4. Emit a Harbor task with environment/Dockerfile (FROM <bootstrap_image>),
     tests/test.sh (the eval script), and solution/patch.diff (gold patch)

Unlike `pr_diff`, this pipeline requires a working Docker image from the
bootstrap phase. `cmd_generate` triggers `ensure_bootstrap()` automatically
when `requires_bootstrap=True`.

----------------------------------------------------------------------------
Acknowledgment
----------------------------------------------------------------------------
This pipeline mirrors the data-collection + validation approach of:

  SWE-bench: Can Language Models Resolve Real-world Github Issues?
  (Jimenez et al., ICLR '24, arXiv:2310.06770)
  https://github.com/SWE-bench/SWE-bench        (MIT)

  SWE-bench-Live: A Live Benchmark for Issue Resolving
  (Zhang et al., NIPS '25, arXiv:2505.23419)
  https://github.com/microsoft/SWE-bench-Live   (MIT)

We adapt the patch-split heuristic (collect/utils.py:extract_patches),
the eval-script structure (harness/test_spec/utils.py:make_eval_script_list_common),
and the F2P/P2P grading semantics (harness/grading.py). No code is copied;
we don't depend on the `swebench` PyPI package.

Released under Apache-2.0 along with the rest of Repo2RLEnv.
----------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, TYPE_CHECKING

if TYPE_CHECKING:
    from repo2rlenv.bootstrap.docker import DockerSandbox
    from repo2rlenv.pipelines.pr_runtime_validate import ValidationOutcome

from pydantic import BaseModel, Field

from repo2rlenv import github
from repo2rlenv.auth import resolve_repo_token
from repo2rlenv.bootstrap.spec import BootstrapResult
from repo2rlenv.emitter.harbor import HarborTask, write_harbor_task
from repo2rlenv.github import GitHubError, PullRequestSummary
from repo2rlenv.gitlab import GitLabError
from repo2rlenv.llm import complete
from repo2rlenv.pipelines._env_guard import (
    egress_guard_compose,
    fix_source_hosts,
    git_history_scrub,
)
from repo2rlenv.pipelines.base import PipelineResult
from repo2rlenv.provider import provider_for
from repo2rlenv.sources import Capability
from repo2rlenv.spec.input import GenerationInput, LLMSpec, PipelineName
from repo2rlenv.spec.options import PRRuntimeOptions

logger = logging.getLogger(__name__)

# PR/MR data calls dispatch by source (GitHub vs GitLab); either can raise.
_PROVIDER_ERRORS = (GitHubError, GitLabError)


_CLOSES_RE = re.compile(r"\b(?:closes|fixes|resolves)\s+#\d+\b", re.IGNORECASE)

# `git clean -fdx` excludes. Keep language dependency/build dirs so resetting
# to a PR's base_commit doesn't wipe installed deps the test suite needs:
#   node_modules (npm/yarn) · target (cargo) · vendor (go) · .venv/venv/.tox
#   (python) · .gradle/build (jvm). Without these, Node/Rust repos would yield
#   ZERO tasks because the suite can't even import its deps after the clean.
_GIT_CLEAN_EXCLUDES = (
    "-e .venv -e venv -e __pycache__ -e .tox "
    "-e node_modules -e target -e vendor -e .gradle -e .next -e .pytest_cache"
)

# Which issue does this PR close? Used to source the problem statement from
# the issue (bug report) instead of the PR body (fix description).
# Matches `Fixes #123`, `Closes #123`, AND the markdown-link form
# `fixes [#123](url)` that PR authors commonly use.
_LINKED_ISSUE_RE = re.compile(
    r"\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s+\[?#(\d+)", re.IGNORECASE
)

# Boilerplate sections that bloat a PR body without describing the problem,
# OR leak the solution. We drop everything from the first such header on.
# Crucially this includes test sections: a "Tests added / updated" block
# names the exact FAIL_TO_PASS tests that grade the task (an eval leak).
_BODY_NOISE_HEADER_RE = re.compile(
    r"^\s*#{0,6}\s*"
    r"(?:checklist|change\s*log|changelog|release\s*notes?|how\s+to\s+test|"
    r"test\s*plan|tests?\s+added(?:\s*/?\s*updated)?|tests?\s+added\s+or\s+updated|"
    r"testing|types?\s+of\s+changes?|pr\s+checklist|reviewer\s+notes?)\s*:?\s*$",
    re.IGNORECASE,
)

# Solution-leak patterns stripped from the problem statement before use.
_LEAK_PATTERNS: tuple[re.Pattern[str], ...] = (
    # "cherry-pick c3535905", "backport commit abc1234", "commit deadbeef1234"
    re.compile(
        r"\b(?:cherry[-\s]?pick(?:ed|ing|s)?|back[-\s]?port(?:ed|ing|s)?|commit)\b"
        r"[^\n]*?\b[0-9a-f]{7,40}\b",
        re.IGNORECASE,
    ),
    # Bare full 40-char SHAs (almost always a git ref pointing at the fix)
    re.compile(r"\b[0-9a-f]{40}\b"),
    # Markdown links to a github PR/issue/commit (point straight at the fix)
    re.compile(r"\[[^\]]*\]\(https?://github\.com/[^\s)]*?/(?:pull|issues|commit)/[^\s)]*\)"),
    # Bare github PR/issue/commit URLs (incl. redirect.github.com)
    re.compile(r"https?://(?:\w+\.)?github\.com/[^\s)]*?/(?:pull|issues|commit)/\S+"),
    # Closes/Fixes/Resolves/See/Refs #N (and cross-repo owner/repo#N)
    re.compile(
        r"\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?|see|refs?|ref|follow[-\s]?up(?:\s+to)?)\b"
        r"[:\s]*(?:[\w.-]+/[\w.-]+)?#\d+",
        re.IGNORECASE,
    ),
    # Markdown issue/PR refs `[#1234](url)` and bare `[#1234]`
    re.compile(r"\[#\d+\]\([^)]*\)"),
    # Parenthesized PR/issue ref at the end of a commit subject — common
    # "merged via squash" trailer: `Fix X (#1234)`. Strip incl. preceding
    # whitespace so the title cleans up to "Fix X".
    re.compile(r"\s*\(#\d+\)"),
    # Cross-repo issue refs without a closes keyword (`gorilla#739`).
    # Distinct from owner/repo#N (which has a slash) — that's already
    # covered by the closes pattern above when paired with a keyword.
    re.compile(r"\b[a-zA-Z][\w.-]*#\d+\b"),
)


def _linked_issue_number(pr_body: str) -> int | None:
    """Return the issue number this PR closes, if any (`Closes #123`)."""
    m = _LINKED_ISSUE_RE.search(pr_body or "")
    return int(m.group(1)) if m else None


def _strip_info_leak(text: str) -> str:
    """Remove solution-pointing references (SHAs, fix-PR links, #refs)."""
    out = text
    for pat in _LEAK_PATTERNS:
        out = pat.sub("", out)
    return out


# A markdown comment block (PR templates wrap guidance in <!-- ... -->).
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
# Collapse 3+ blank lines down to a single blank line.
_MULTI_BLANK_RE = re.compile(r"\n{3,}")
# Cap the instruction body so a 5000-word PR essay doesn't dominate context.
_MAX_BODY_CHARS = 4000


def _reflow_pr_body(body: str) -> str:
    """Tidy a verbose PR body into a focused problem statement.

    - drop HTML comment blocks (PR-template guidance)
    - drop everything from the first checklist/changelog/template header on
    - collapse runs of blank lines
    - cap length

    Conservative: only trims obvious boilerplate. Never rewrites prose (no
    LLM call here — deterministic + cheap).
    """
    body = _HTML_COMMENT_RE.sub("", body)
    kept: list[str] = []
    for line in body.splitlines():
        if _BODY_NOISE_HEADER_RE.match(line):
            break  # the rest is checklist/changelog noise
        kept.append(line)
    out = _MULTI_BLANK_RE.sub("\n\n", "\n".join(kept)).strip()
    if len(out) > _MAX_BODY_CHARS:
        out = out[:_MAX_BODY_CHARS].rstrip() + "\n\n…(truncated)"
    return out


# Path-component classifier for "is this a test file?".
#
# SWE-bench's heuristic is a substring match on the full path — which over-fires:
# `docs/testing.md`, `src/click/testing.py`, etc. all become "test files".
# We instead match on PATH COMPONENTS (split by /) and explicitly exclude
# documentation paths.
_TEST_DIR_NAMES = {"test", "tests", "testing", "e2e", "__tests__"}
_DOC_PREFIX_DIRS = {"docs", "doc", "documentation", "examples", "example"}
# jest/vitest's default test-file names: `*.test.ts`, `*.spec.tsx`, `*.test.mjs`, ...
_JS_TEST_FILE_RE = re.compile(r"\.(?:test|spec)\.[cm]?[jt]sx?$")


def _path_is_test(path: str) -> bool:
    """True if the file is a real test file, false for docs / src files
    that merely contain a test keyword in their path.

    Rules:
      1. Files under a documentation root (`docs/`, `examples/`, ...) are
         NEVER test files, even if their name contains "test".
      2. Files inside a test directory component (any path part in
         `_TEST_DIR_NAMES`) are test files.
      3. Files with a pytest-style basename (`test_*.py`, `*_test.py`,
         `*_test.go`) or a jest/vitest one (`*.test.ts`, `*.spec.tsx`) are
         test files.
    """
    if not path:
        return False
    parts = [p.lower() for p in path.split("/") if p]
    if not parts:
        return False
    # Rule 1: skip anything under a docs root
    if parts[0] in _DOC_PREFIX_DIRS:
        return False
    # Rule 2: any directory component is a known test dir
    # (excluding the last component, which is the file name)
    for component in parts[:-1]:
        if component in _TEST_DIR_NAMES:
            return True
    # Rule 3: filename-level test markers
    basename = parts[-1]
    return (
        (basename.startswith("test_") and basename.endswith((".py", ".js", ".ts")))
        or basename.endswith(("_test.py", "_test.go"))
        or _JS_TEST_FILE_RE.search(basename) is not None
    )


# Match `diff --git a/<path> b/<path>` block boundaries to split a unified diff
_DIFF_HEADER_RE = re.compile(r"^diff --git a/(\S+) b/(\S+)$", re.MULTILINE)


def split_patch_and_test_patch(unified_diff: str) -> tuple[str, str]:
    """Split a PR's unified diff into (source patch, test patch).

    SWE-bench rule: a file hunk goes into `test_patch` iff its path contains
    one of `test/tests/e2e/testing`; everything else goes into `patch`.

    We walk the diff by `diff --git` markers (one per file in the PR) so we
    keep each file's hunks intact.
    """
    if not unified_diff.strip():
        return "", ""

    # Find each "diff --git a/X b/Y" header and the byte offset where its block starts
    matches = list(_DIFF_HEADER_RE.finditer(unified_diff))
    if not matches:
        # Empty / malformed — return whole thing as patch
        return unified_diff, ""

    patch_parts: list[str] = []
    test_parts: list[str] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(unified_diff)
        block = unified_diff[start:end]
        path_a = m.group(1)
        path_b = m.group(2)
        # If EITHER path looks like a test file (covers renames, new files), it's a test
        if _path_is_test(path_a) or _path_is_test(path_b):
            test_parts.append(block)
        else:
            patch_parts.append(block)

    return "".join(patch_parts), "".join(test_parts)


def _build_instruction(
    pr: PullRequestSummary,
    owner: str = "",
    name: str = "",
    *,
    token: str | None = None,
    provider=github,
    hard_mode: bool = False,
) -> str:
    """Build the task instruction (problem statement).

    Sources the problem from the **linked issue** (the bug *report*) when the
    PR closes one, falling back to the PR body otherwise. The PR body is
    written by the fixer and routinely leaks the solution (commit SHAs to
    cherry-pick, the fix approach, the names of the grading tests); the issue
    describes the *symptom*, which is what we want the agent to work from.
    This mirrors SWE-bench, which builds problem statements from issue text.

    Whatever text we use is then run through the info-leak strip + reflow so
    stray fix-PR links / #refs / SHAs / test-section noise don't leak through.
    """
    if hard_mode:
        body = (pr.body or "").strip()
        return f"{pr.title}\n\n{body}".strip() if body else pr.title.strip()

    title = pr.title
    body = pr.body or ""

    issue_num = _linked_issue_number(pr.body or "")
    if issue_num is not None and owner and name:
        fetched = provider.fetch_issue(owner, name, issue_num, token=token)
        if fetched:
            i_title, i_body = fetched
            title = i_title or pr.title
            body = i_body or body

    body = _CLOSES_RE.sub("", body)
    body = _strip_info_leak(body)
    body = _reflow_pr_body(body).strip()
    if not body:
        body = "(no description provided in source issue/PR)"
    title = _strip_info_leak(title).strip() or pr.title
    return (
        f"# Issue\n\n"
        f"**Title:** {title}\n\n"
        f"## Description\n\n"
        f"{body}\n\n"
        f"## Task\n\n"
        f"Modify the repository so that the issue described above is resolved. "
        f"The task's test suite verifies your patch by applying it on top of "
        f"the base commit `{pr.base_sha[:12]}` and running the modified tests."
    )


_RECIPES_SWE_NEXT = Path(__file__).resolve().parent / "recipes" / "swe_next"


def _load_swe_next_assets() -> tuple[str, str]:
    guide = (_RECIPES_SWE_NEXT / "instruction_prompt.md").read_text(encoding="utf-8")
    examples = (_RECIPES_SWE_NEXT / "issue_examples.json").read_text(encoding="utf-8")
    return guide, examples


def synthesize_task_instruction(
    *,
    pr_title: str,
    pr_body: str,
    base_commit: str,
    fail_to_pass: list[str],
    pre_log: str = "",
    issue_title: str = "",
    issue_body: str = "",
    patch: str = "",
    test_patch: str = "",
    llm_spec: LLMSpec,
) -> str:
    guide, examples = _load_swe_next_assets()
    system_prompt = (
        f"{guide}\n\n"
        f"--- IN-CONTEXT EXAMPLES ---\n{examples}\n\n"
        f"Output format:\n"
        f"Title: <Title>\n\n"
        f"Description:\n<Description>\n\n"
        f"Currently on baseline:\n```python\n...\n```\n\n"
        f"Requirements:\n...\n\n"
        f"Actual behavior:\n...\n\n"
        f"Ground the description in the observed test failures, diffs, and runtime tracebacks.\n"
        f"The PR diff, solution, and test names are strictly private authoring context: DO NOT reveal any code fix, "
        f"commit SHAs, PR numbers, or internal test file names.\n"
        f"IMPORTANT: If new public functions/classes are added by this PR, infer their signatures and return types "
        f"directly from the Test Patch (Verifier) and specify them under Requirements. DO NOT import unbuilt symbols "
        f"in the 'Currently on baseline' code snippet."
    )

    user_parts: list[str] = [
        f"# PR Title\n{pr_title.strip()}",
    ]
    if pr_body.strip():
        user_parts.append(f"# PR Description / Context\n{pr_body.strip()}")
    if issue_title.strip() or issue_body.strip():
        linked_parts = []
        if issue_title.strip():
            linked_parts.append(f"Title: {issue_title.strip()}")
        if issue_body.strip():
            linked_parts.append(f"Body:\n{issue_body.strip()}")
        user_parts.append(f"# Linked Issue\n" + "\n\n".join(linked_parts))
    if patch.strip():
        patch_snippet = patch.strip()
        if len(patch_snippet) > 8000:
            patch_snippet = patch_snippet[:8000] + "\n...[truncated]..."
        user_parts.append(
            f"# Solution (Gold Patch - Private Authoring Context)\n```diff\n{patch_snippet}\n```"
        )
    if test_patch.strip():
        test_patch_snippet = test_patch.strip()
        if len(test_patch_snippet) > 8000:
            test_patch_snippet = test_patch_snippet[:8000] + "\n...[truncated]..."
        user_parts.append(
            f"# Test Patch (Verifier Tests - Private Authoring Context)\n```diff\n{test_patch_snippet}\n```"
        )
    if fail_to_pass:
        user_parts.append(
            "# Failing Test Targets\n"
            + "\n".join(f"- {t}" for t in fail_to_pass)
        )
    if pre_log.strip():
        log_snippet = pre_log.strip()
        if len(log_snippet) > 8000:
            log_snippet = log_snippet[-8000:]
        user_parts.append(f"# Baseline Test Execution Log (Pre-Fix Output)\n```\n{log_snippet}\n```")
    user_prompt = "\n\n".join(user_parts)

    synth_llm = llm_spec.model_copy(update={"reasoning_effort": None})
    response = complete(
        synth_llm,
        system=system_prompt,
        user=user_prompt,
        max_tokens=4096,
    )
    content = response.content.strip()
    if "[ISSUE]" in content and "[/ISSUE]" in content:
        content = content.split("[ISSUE]")[1].split("[/ISSUE]")[0].strip()
    elif "[ISSUE]" in content:
        content = content.split("[ISSUE]")[1].strip()

    # Clean up formatting artifacts if present
    if content.startswith("# Title:"):
        content = content[2:].lstrip()
    elif content.startswith("hTitle:"):
        content = "Title:" + content[7:]

    return content


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))


# PR titles that signal a non-bug chore — not a meaningful SWE task, and their
# bodies routinely leak the fix (commit SHA to cherry-pick, the source PR).
_NON_BUG_TITLE_RE = re.compile(
    r"\b(?:back[-\s]?port|cherry[-\s]?pick|revert|"
    r"bump|release|changelog|forward[-\s]?merge|"
    r"prepare\s+(?:for\s+)?release|version\s+bump|re-?sync)\b"
    # merge-forward / branch-sync PRs: "merge branch", "merge stable into main",
    # "merge X into Y", "sync stable", "sync main", "merge master" — these are
    # broad branch syncs, not focused bug fixes, and produce huge noisy diffs.
    r"|\bmerge\b[^\n]*\binto\b"
    r"|\bmerge\s+(?:branch|stable|main|master|develop|upstream|release)\b"
    r"|\bsync\s+(?:stable|main|master|branch|develop|upstream)\b",
    re.IGNORECASE,
)


def _is_non_bug_pr(title: str) -> bool:
    """True if the PR title marks a backport / cherry-pick / release / revert."""
    return bool(_NON_BUG_TITLE_RE.search(title or ""))


def _verifier_source() -> str:
    """Read the standalone graded verifier's source for base64 embedding.

    Mirrors pr_diff: the verifier (`_pr_runtime_verifier.py`) runs inside
    the task container, so we read its source at gen time and bake it into
    `tests/test.sh` as a base64 blob.
    """
    return (Path(__file__).parent / "_pr_runtime_verifier.py").read_text(encoding="utf-8")


def build_environment_dockerfile(bootstrap_image: str, base_commit: str) -> str:
    """Build the fallback per-task environment/Dockerfile."""
    return (
        f"# Auto-generated by Repo2RLEnv pr_swegemma_runtime\n"
        f"FROM {bootstrap_image}\n"
        f"WORKDIR /workspace\n"
        f"RUN git config --global --add safe.directory /workspace \\\n"
        f"    && git reset --hard {base_commit} && git clean -fdx {_GIT_CLEAN_EXCLUDES}\n"
        + git_history_scrub(base_commit)
    )


def _path_prelude_for_language(language: str | None) -> str:
    """Shell snippet that prepends common toolchain dirs to $PATH.

    The bootstrap agent often installs language toolchains (Go, Rust,
    Node) into well-known paths (`/usr/local/go/bin`, `~/.cargo/bin`,
    nvm dirs) but doesn't always persist a corresponding `export PATH`
    to a shell init file. When Harbor's verifier runs `bash test.sh` in
    a non-interactive shell, those binaries vanish from PATH → exit 127
    on `go test` / `cargo test` / `node` → false-negative reward 0.

    The fix at emission time: prepend the known install locations for
    the bootstrap-detected language so the verifier shell always finds
    the runner binary. Missing dirs are no-ops; the cost is one extra
    line in test.sh.
    """
    extras = {
        "go": ["/usr/local/go/bin", "$HOME/go/bin"],
        "rust": ["$HOME/.cargo/bin"],
        "node": ["/usr/local/lib/node_modules/.bin", "$HOME/.nvm/versions/node/*/bin"],
        "java": ["/usr/lib/jvm/default-java/bin"],
    }
    dirs = extras.get((language or "").lower(), [])
    if not dirs:
        return ""
    joined = ":".join(dirs)
    return f'export PATH="{joined}:$PATH"\n'


def build_eval_script(
    base_commit: str,
    test_patch: str,
    test_cmds: list[str],
    *,
    language: str | None = None,
    fail_to_pass: list[str] | None = None,
    pass_to_pass: list[str] | None = None,
) -> str:
    """Build the `tests/test.sh` content that Harbor runs after the model patch.

    Adapted from SWE-bench's `harness/test_spec/utils.py:make_eval_script_list_common`.
    The flow:
      1. cd /workspace + mark safe.directory (for non-root git operations)
      2. Prepend known toolchain paths for the detected language (compensates
         for bootstrap agents that install Go/Rust/Node outside /usr/bin
         without exporting PATH in any persisted shell init file)
      3. Reset test files to base_commit (so re-running stays clean) — this is
         the anti-tamper guard: the agent CANNOT pass by editing/deleting the
         tests, because we restore them to base and re-apply the test_patch
      4. Apply the test_patch (via heredoc + git apply --reject)
      5. Run test_cmds, capturing stdout+stderr to a log file
      6. Score the reward:
         - GRADED (default, when fail_to_pass is provided): bake the standalone
           graded verifier + the F2P/P2P test-name lists, parse the captured
           log, and write reward = f2p_rate * p2p_rate to reward.txt (+ full
           breakdown incl. the strict SWE-bench ``resolved`` bool to reward-details.json)
         - BINARY (fallback, e.g. ``skip_validation`` with no F2P known): write
           1.0 if the suite exited 0 else 0.0
      7. Reset test files again on the way out

    The model's predicted patch is applied by Harbor *before* this script runs.
    """
    test_files = _files_in_patch(test_patch)
    heredoc = "EOF_R2E_TEST_PATCH"
    reset = (
        f"git checkout {base_commit} -- {' '.join(test_files)}"
        if test_files
        else "echo 'no test files to reset'"
    )
    apply = f"git apply --verbose --reject - <<'{heredoc}'\n{test_patch}\n{heredoc}"
    test_block = " && ".join(test_cmds) if test_cmds else "echo 'no test_cmds configured'"
    path_prelude = _path_prelude_for_language(language)

    # Fail CLOSED if the hidden test_patch doesn't apply: an agent that edited
    # tests or the file layout could make `git apply` fail, and we must NOT
    # then score against stale/native tests. Write reward 0 + a clear status
    # and stop. (Reset tolerates test files absent at base — that's `|| true`.)
    apply_guard = (
        ""
        if not test_patch.strip()
        else (
            f"{apply}\n"
            "R2E_APPLY_RC=$?\n"
            'if [ "$R2E_APPLY_RC" -ne 0 ]; then\n'
            '  echo "0.000000" > /logs/verifier/reward.txt\n'
            "  printf '%s' "
            '\'{"reward": 0.0, "resolved": false, "parse_status": '
            '"test_patch_apply_failed"}\' > /logs/verifier/reward-details.json\n'
            '  echo "R2E: test_patch failed to apply (rc=$R2E_APPLY_RC) — failing closed" >&2\n'
            "  exit 0\n"
            "fi\n"
        )
    )
    head = (
        "#!/bin/bash\n"
        "set -uxo pipefail\n"
        "for i in $(seq 1 120); do\n"
        "  if [ -f /ready ]; then break; fi\n"
        "  sleep 0.5\n"
        "done\n"
        f"{path_prelude}"  # may be empty
        # Harbor mounts the task's tests/ dir at /tests; resolve it so we can
        # read the sibling verifier.py + f2p.json + p2p.json artifacts.
        'SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"\n'
        "cd /workspace\n"
        "git config --global --add safe.directory /workspace\n"
        "mkdir -p /logs/verifier\n"
        f"{reset} || true\n"  # tolerate test files that didn't exist at base
        f"{apply_guard}"
        # Capture the suite output so the verifier can parse per-test status.
        # `( ... )` subshell + redirect keeps the whole `a && b` block's output;
        # pipefail preserves the real test exit code.
        f"( {test_block} ) > /logs/verifier/test_output.log 2>&1\n"
        "TEST_EXIT_CODE=$?\n"
        "cat /logs/verifier/test_output.log\n"
    )

    if not fail_to_pass:
        # No F2P oracle available (e.g. --skip-validation). Fall back to the
        # binary exit-code reward.
        return (
            head + '[ "$TEST_EXIT_CODE" -eq 0 ] && echo "1.0" > /logs/verifier/reward.txt '
            '|| echo "0.0" > /logs/verifier/reward.txt\n'
            f"{reset} || true\n"
            "exit 0\n"
        )

    # Graded path: a thin orchestrator that runs the verifier + F2P/P2P lists
    # shipped as PLAIN task artifacts (tests/verifier.py, tests/f2p.json,
    # tests/p2p.json — written by `_runtime_aux_files`). No base64 blobs:
    # the task semantics are inspectable, and test.sh stays small regardless
    # of how many F2P/P2P tests a task has.
    cmds_str = " ".join(test_cmds).replace("'", "'\\''")
    return (
        head + 'python3 "$SCRIPT_DIR/verifier.py" '
        "--log /logs/verifier/test_output.log "
        '--f2p "$SCRIPT_DIR/f2p.json" --p2p "$SCRIPT_DIR/p2p.json" '
        f"--test-cmds '{cmds_str}' --exit-code \"$TEST_EXIT_CODE\" "
        "--out-dir /logs/verifier || "
        # If python3 is somehow unavailable, never leave reward.txt empty.
        '{ [ "$TEST_EXIT_CODE" -eq 0 ] && echo "1.0" > /logs/verifier/reward.txt '
        '|| echo "0.0" > /logs/verifier/reward.txt; }\n'
        f"{reset} || true\n"  # cleanup; failure here doesn't change verdict
        "exit 0\n"  # reward.txt is the verdict, not the bash exit code
    )


def _runtime_aux_files(fail_to_pass: list[str], pass_to_pass: list[str]) -> dict[str, str]:
    """The plain task artifacts the graded test.sh reads from /tests.

    Shipping these as files (vs base64 inside test.sh) keeps task semantics
    inspectable and test.sh small. Harbor exposes tests/ at /tests in the
    container.
    """
    return {
        "tests/verifier.py": _verifier_source(),
        "tests/f2p.json": json.dumps(fail_to_pass, indent=2),
        "tests/p2p.json": json.dumps(pass_to_pass or [], indent=2),
    }


def _files_in_patch(unified_diff: str) -> list[str]:
    """Extract the unique 'b/' file paths touched by a unified diff."""
    if not unified_diff.strip():
        return []
    seen: list[str] = []
    for m in _DIFF_HEADER_RE.finditer(unified_diff):
        b = m.group(2)
        if b not in seen:
            seen.append(b)
    return seen


# Lines in a test_patch that introduce a new test function/class.
# Python: `+def test_foo(...)`, `+    def test_bar(...)`, `+class TestX`
# JS/TS:  `+it(`, `+test(`, `+describe(` (not currently filtered)
# Go:     `+func TestFoo(`
_NEW_TEST_FUNC_RE = re.compile(
    r"^\+\s*(?:def\s+test_\w+|class\s+\w*[Tt]est\w*|func\s+Test\w+|it\s*\(|test\s*\(|describe\s*\()",
)


def _diff_loc_changed(unified_diff: str) -> int:
    """Count real +/- lines in a unified diff (excludes +++/--- file markers)."""
    n = 0
    for line in (unified_diff or "").splitlines():
        if (line.startswith("+") or line.startswith("-")) and not line.startswith(("+++ ", "--- ")):
            n += 1
    return n


def _difficulty_bucket(f2p_count: int, loc_changed: int) -> str:
    """Coarse difficulty from oracle size — lets consumers slice train/eval.

    Combines the fix size (LOC) and how many distinct behaviours must be
    restored (F2P count). Mirrors Arc 1's pr_diff buckets on LOC.
    """
    if loc_changed <= 5 and f2p_count <= 1:
        return "trivial"
    if loc_changed <= 20:
        return "small"
    if loc_changed <= 80:
        return "medium"
    return "large"


def _count_new_test_funcs(test_patch: str) -> int:
    """Count new test-function definitions added in a unified diff.

    Used to filter out PRs whose test_patch is comment-only or docstring-only
    (cosmetic changes that can't produce a FAIL_TO_PASS oracle).
    """
    if not test_patch.strip():
        return 0
    return sum(1 for line in test_patch.splitlines() if _NEW_TEST_FUNC_RE.match(line))


_PYTEST_COMMAND_RE = re.compile(r"\b(?:pytest|py[.]test)\b")


def normalize_test_cmds_for_runtime(test_cmds: list[str]) -> list[str]:
    """Adapt bootstrap-recorded test commands for actual per-PR execution.

    Bootstrap prefers fast/tolerant commands (e.g. `pytest --collect-only`)
    so it can declare success without running every test. For pr_runtime,
    we need commands that *run* tests and emit per-test pass/fail lines
    that our parsers can read.

    Transforms (per runner):
      pytest / py.test:
        - Drop `--collect-only` / `--co` so pytest actually runs tests
        - Drop `-q` / `--quiet`: suppresses per-test names; cancels `-v` in pytest 9
        - Add `-v` if no verbosity flag is present
      python unittest / Django:
        - Add a verbosity flag when none is present, since these runners
          print only dots at the default level: `-v` for unittest, `-v 2`
          for `manage.py test`, and `--verbosity 2` for Django's in-repo
          `tests/runtests.py`, which takes no `-v` count.
      go test:
        - Add `-v` if missing (default `go test` doesn't print --- PASS lines)
      cargo test:
        - Default output is already parseable; no transform needed
      cargo nextest run:
        - Add `--status-level pass` (or raise a lower one) and `--color never`
      jest / npm test:
        - Add `--verbose` if not present, so per-test ✓/✕ lines are emitted
        - Some configs swallow stdout via `--silent`; we strip that
      mocha:
        - Use `--reporter spec` instead: mocha has no `--verbose` option and
          reads the next word as its value, dropping a test file appended later
      vitest:
        - Use `--reporter=verbose` instead: vitest rejects `--verbose`, and its
          default reporter collapses fully passing files to one summary line
    """
    out: list[str] = []
    for cmd in test_cmds:
        cleaned = cmd

        # Strip shell pipes / redirects / tail-truncators that bootstrap agents
        # sometimes append (e.g. `pytest -q 2>&1 | head -50`) so we capture only
        # the test runner invocation. If we keep them, `targeted_test_cmds_for_pr`
        # appends test files AFTER the pipe → broken command.
        # `[^|]*` swallows whatever flags follow `head`/`tail` (`-50`, `-n 100`, etc.)
        # without crossing into another piped command.
        cleaned = re.sub(r"\s*\|\s*(?:head|tail)\s*[^|]*$", "", cleaned)
        cleaned = re.sub(r"\s*2>&1\b", "", cleaned)
        cleaned = re.sub(r"\s*&?>\s*/dev/null\b", "", cleaned)
        cleaned = cleaned.rstrip(" |&")

        # --- pytest ---
        if _PYTEST_COMMAND_RE.search(cleaned):
            cleaned = re.sub(r"\s+--collect-only\b", "", cleaned)
            cleaned = re.sub(r"\s+--co\b", "", cleaned)  # pytest's short form
            # Strip -q/--quiet: it suppresses per-test names that the log parser needs.
            # -q and -v cancel each other in pytest 9 (verbosity counter), so -q must go.
            cleaned = re.sub(r"\s+(?:-q|--quiet)\b", "", cleaned)
            if not re.search(r"\s-v\b|\s--verbose\b|-vv\b", cleaned):
                cleaned = cleaned.rstrip() + " -v"

        # --- python unittest / Django (manage.py test, tests/runtests.py) ---
        elif re.search(r"\bunittest\b|manage\.py\s+test\b|runtests\.py", cleaned):
            if not re.search(r"\s-v\b|\s--verbose\b|\s--verbosity\b|-vv\b", cleaned):
                if re.search(r"runtests\.py", cleaned):
                    flag = " --verbosity 2"
                elif re.search(r"manage\.py\s+test\b", cleaned):
                    flag = " -v 2"  # Django's runner counts verbosity
                else:
                    flag = " -v"  # unittest's own flag
                cleaned = cleaned.rstrip() + flag

        # --- go test ---
        elif re.search(r"\bgo\s+test\b", cleaned):
            if not re.search(r"\s-v\b", cleaned):
                # Insert -v right after `go test`; positional args go after
                cleaned = re.sub(r"\bgo\s+test\b", "go test -v", cleaned, count=1)

        # --- cargo nextest ---
        elif re.search(r"\bcargo\s+nextest\s+run\b", cleaned):
            # A repo's nextest config can lower `status-level` and hide the
            # PASS lines that FAIL_TO_PASS needs, so ask for them explicitly.
            # Flags go right after `run`: anything after `--` goes to the tests.
            cleaned = re.sub(
                r"\s+--status-level(?:=|\s+)(?:none|fail|retry|slow|leak)\b", "", cleaned
            )
            extra = ""
            if not re.search(r"\s--status-level\b", cleaned):
                extra += " --status-level pass"
            if not re.search(r"\s--color\b", cleaned):
                extra += " --color never"
            cleaned = re.sub(r"\bcargo\s+nextest\s+run\b", r"\g<0>" + extra, cleaned, count=1)

        # --- cargo test ---
        elif re.search(r"\bcargo\s+test\b", cleaned):
            # `cargo test` already prints `test NAME ... ok/FAILED/ignored`
            # by default — no transformation needed. If a user passed
            # `-q`, the per-test lines disappear; strip it.
            cleaned = re.sub(r"\s+(?:-q|--quiet)\b", "", cleaned)

        # --- jest / npm test / yarn test / pnpm test ---
        elif re.search(r"\b(?:jest|mocha|vitest|npm\s+test|yarn\s+test|pnpm\s+test)\b", cleaned):
            cleaned = re.sub(r"\s+--silent\b", "", cleaned)
            # Add a per-test reporter if the cmd is the runner itself (skip
            # wrappers where flags need to go after `--`)
            if re.search(r"\bvitest\b", cleaned):
                # `vitest --verbose` exits with `CACError: Unknown option`.
                cleaned = re.sub(r"\s+--verbose\b", "", cleaned)
                if not re.search(r"\s--reporter\b", cleaned):
                    cleaned = cleaned.rstrip() + " --reporter=verbose"
            elif re.search(r"\bmocha\b", cleaned):
                # mocha has no `--verbose`: it reads the next word as the flag's
                # value, which swallows a test file appended after it.
                cleaned = re.sub(r"\s+--verbose\b", "", cleaned)
                if not re.search(r"\s(?:--reporter|-R)\b", cleaned):
                    cleaned = cleaned.rstrip() + " --reporter spec"
            elif re.search(r"\bjest\b", cleaned) and not re.search(
                r"\s--verbose\b|\s--reporter\b", cleaned
            ):
                cleaned = cleaned.rstrip() + " --verbose"

        out.append(cleaned.strip())
    return out


# Pytest-style file extensions we know how to target. Anything else triggers
# the fallback to "run the whole suite".
_PYTEST_TARGETABLE_EXT = (".py",)
_JEST_TARGETABLE_EXT = (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs")


def _go_packages_from_test_files(test_files: list[str]) -> list[str]:
    """Map `pkg/foo/bar_test.go` → `./pkg/foo` for Go's package-path CLI."""
    pkgs: list[str] = []
    for f in test_files:
        if not f.endswith("_test.go"):
            continue
        # Directory of the test file, prefixed with ./ for `go test` package syntax
        parts = f.rsplit("/", 1)
        pkg = "./" + parts[0] if len(parts) == 2 else "./"
        if pkg not in pkgs:
            pkgs.append(pkg)
    return pkgs


def targeted_test_cmds_for_pr(test_cmds: list[str], test_files: list[str]) -> list[str]:
    """Limit the test invocation to the file paths the PR's test_patch touches.

    Running the whole suite on every PR is 10-50× slower than running only
    the files the PR cares about. SWE-bench-Live's harness does the same.

    Per-runner rules:
      pytest / py.test: append the changed .py test files as positional args
      jest:    append the changed .js/.ts test files as positional args
      go test: replace `./...` (or trailing nothing) with the package
               directories containing changed `*_test.go` files
      cargo:   no targeting — Rust's filter is name-substring, not file
               (we'd need to introspect test names; whole-suite is fine)

    Skips if the cmd already has a positional path arg.
    """
    if not test_files:
        return test_cmds

    py_files = [
        f
        for f in test_files
        if f.endswith(_PYTEST_TARGETABLE_EXT)
        and (Path(f).name.startswith("test_") or Path(f).name.endswith("_test.py"))
    ]
    js_files = [f for f in test_files if f.endswith(_JEST_TARGETABLE_EXT)]
    go_pkgs = _go_packages_from_test_files(test_files)

    out: list[str] = []
    for cmd in test_cmds:
        # --- pytest ---
        if _PYTEST_COMMAND_RE.search(cmd) and py_files:
            tokens = cmd.split()
            pytest_idx = next(
                (
                    i
                    for i, t in enumerate(tokens)
                    if t in {"pytest", "py.test"} or t.endswith(("/pytest", "/py.test"))
                ),
                -1,
            )
            if pytest_idx >= 0:
                tail = tokens[pytest_idx + 1 :]
                has_path_arg = any(
                    not t.startswith("-") and (t.endswith(".py") or "/" in t) for t in tail
                )
                if not has_path_arg:
                    cmd = cmd.rstrip() + " " + " ".join(py_files)

        # --- go test ---
        elif re.search(r"\bgo\s+test\b", cmd) and go_pkgs:
            # Replace `./...` with the targeted packages; if neither is present,
            # append the packages
            if "./..." in cmd:
                cmd = cmd.replace("./...", " ".join(go_pkgs))
            elif not re.search(r"\b\./\S+\b", cmd):
                cmd = cmd.rstrip() + " " + " ".join(go_pkgs)

        # --- jest / npx jest / mocha / vitest ---
        elif re.search(r"\b(?:jest|mocha|vitest)\b", cmd) and js_files:
            tokens = cmd.split()
            # If a positional file path is already present, don't double-up
            has_path = any(
                not t.startswith("-") and (t.endswith(_JEST_TARGETABLE_EXT) or "/" in t)
                for t in tokens[1:]
            )
            if not has_path:
                cmd = cmd.rstrip() + " " + " ".join(js_files)

        out.append(cmd)
    return out


class PRSWEGemmaRuntimePipeline:
    """Sandbox-verified PR mining for SWEGemma. Implements the `Pipeline` Protocol."""

    name: ClassVar[PipelineName] = PipelineName.PR_SWEGEMMA_RUNTIME
    required_capabilities: ClassVar[frozenset[Capability]] = frozenset({Capability.PULL_REQUESTS})
    requires_bootstrap: ClassVar[bool] = True
    experimental: ClassVar[bool] = False  # stable

    def __init__(
        self,
        input: GenerationInput,
        options: PRRuntimeOptions,
        bootstrap: BootstrapResult | None = None,
    ):
        if bootstrap is None:
            raise RuntimeError(
                "pr_swegemma_runtime requires a BootstrapResult (set requires_bootstrap=True "
                "and let cmd_generate trigger it, or pass one explicitly)"
            )
        self.input = input
        self.options = options
        self.bootstrap = bootstrap
        self._progress_cb = None
        self._token: str | None = None
        self._provider = github

    def set_progress_callback(self, cb) -> None:
        self._progress_cb = cb

    def _emit_progress(self, name: str, outcome: str, reason: str = "") -> None:
        if self._progress_cb is not None:
            try:
                self._progress_cb(name=name, outcome=outcome, reason=reason)
            except Exception as exc:
                logger.debug("progress callback failed: %s", exc)

    # ----- run loop -----------------------------------------------------------

    def run(self, out_dir: Path) -> PipelineResult:
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed

        out_dir.mkdir(parents=True, exist_ok=True)

        token = resolve_repo_token(self.input.repo, self.input.auth)
        self._token = token  # reused by _build_task to fetch linked-issue text
        self._provider = provider_for(self.input.repo)  # GitHub vs GitLab dispatch
        if self.input.repo.access == "private" and not token:
            raise RuntimeError(
                "private repo specified but no token resolved. "
                "Run `gh auth login` / set GITHUB_TOKEN (GitHub) or GITLAB_TOKEN (GitLab)."
            )

        owner, name = self.input.repo.owner_name
        if self.options.prs:
            logger.info("fetching %d specific PRs for %s/%s", len(self.options.prs), owner, name)
            prs: list[PullRequestSummary] = []
            for num in self.options.prs:
                try:
                    summary = self._provider.fetch_pr_summary(owner, name, num, token=token)
                except _PROVIDER_ERRORS as exc:
                    logger.warning("PR #%d: fetch failed: %s", num, exc)
                    summary = None
                if summary is not None:
                    prs.append(summary)
                else:
                    logger.warning("PR #%d: could not fetch summary, skipping", num)
        else:
            logger.info("listing merged PRs for %s/%s (limit=%d)", owner, name, self.options.limit)
            try:
                prs = self._provider.list_merged_prs(
                    owner,
                    name,
                    limit=self.options.limit,
                    since=self.options.since,
                    until=self.options.until,
                    skip_drafts=self.options.skip_drafts,
                    token=token,
                )
            except _PROVIDER_ERRORS as exc:
                raise RuntimeError(f"failed to list PRs: {exc}") from exc

        class _WorkerState(threading.local):
            sandbox: DockerSandbox | None = None

        thread_local = _WorkerState()
        sandboxes_lock = threading.Lock()
        active_sandboxes: list[DockerSandbox] = []

        def get_thread_sandbox() -> DockerSandbox:
            sb = thread_local.sandbox
            if sb is None:
                sb = self._start_validation_sandbox()
                thread_local.sandbox = sb
                with sandboxes_lock:
                    active_sandboxes.append(sb)
            return sb

        skip_reasons: dict[str, int] = {}
        emitted = 0
        results_lock = threading.Lock()

        def process_candidate(pr: PullRequestSummary) -> tuple[str, str, str]:
            pr_label = f"{owner}/{name}#{pr.number}"

            # Pre-validation skip filters
            reason = self._pre_filter(pr)
            if reason:
                return (pr_label, "skip", reason)

            # Fetch diff
            try:
                diff = self._provider.fetch_pr_diff(owner, name, pr.number, token=token)
            except _PROVIDER_ERRORS as exc:
                logger.warning("PR #%d: diff fetch failed: %s", pr.number, exc)
                return (pr_label, "error", "diff_fetch_failed")

            patch, test_patch = split_patch_and_test_patch(diff)
            if not patch.strip():
                return (pr_label, "skip", "empty_source_patch")
            if not test_patch.strip():
                return (pr_label, "skip", "no_test_patch")

            # Structural quality filters (cheap, run before validation):
            # - CI-only PRs: source patch is 100% under .github/
            # - No new test functions: test_patch only edits comments/docstrings
            structural_reason = self._structural_quality_filter(patch, test_patch)
            if structural_reason:
                return (pr_label, "skip", structural_reason)

            # Lite-style structural filters
            lite_reason = self._lite_filter(pr, patch)
            if lite_reason:
                return (pr_label, "skip", lite_reason)

            # Validation (optional via skip_validation)
            fail_to_pass: list[str] = []
            pass_to_pass: list[str] = []
            validation_status = "skipped"

            if not self.options.skip_validation:
                sandbox = get_thread_sandbox()
                from repo2rlenv.pipelines.pr_runtime_validate import validate_pr

                targeted_cmds = targeted_test_cmds_for_pr(
                    normalize_test_cmds_for_runtime(self.bootstrap.test_cmds),
                    _files_in_patch(test_patch),
                )
                outcome = validate_pr(
                    sandbox=sandbox,
                    base_commit=pr.base_sha,
                    patch=patch,
                    test_patch=test_patch,
                    test_cmds=targeted_cmds,
                    rebuild_cmds=self.bootstrap.rebuild_cmds if self.bootstrap.language.value != "python" else None,
                    language=self.bootstrap.language.value,
                    timeout=self.options.validation_timeout_sec,
                )
                fail_to_pass = outcome.fail_to_pass
                pass_to_pass = outcome.pass_to_pass
                validation_status = outcome.status

                if (
                    self.options.require_fail_to_pass
                    and len(fail_to_pass) < self.options.min_fail_to_pass
                ):
                    return (pr_label, "skip", outcome.reason or "no_fail_to_pass")

            # Emit the Harbor task
            task = self._build_task(
                pr,
                patch,
                test_patch,
                fail_to_pass=fail_to_pass,
                pass_to_pass=pass_to_pass,
                validation_status=validation_status,
            )
            write_harbor_task(task, out_dir)
            logger.info(
                "emitted task %s (F2P=%d, P2P=%d)",
                task.name,
                len(fail_to_pass),
                len(pass_to_pass),
            )
            return (task.name, "emit", "")

        try:
            max_workers = max(1, self.options.concurrency)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(process_candidate, pr) for pr in prs]
                for future in as_completed(futures):
                    label, action, reason = future.result()
                    with results_lock:
                        if action == "emit":
                            emitted += 1
                            self._emit_progress(label, "emit")
                        elif action == "skip":
                            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                            self._emit_progress(label, "skip", reason)
                        elif action == "error":
                            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                            self._emit_progress(label, "error", reason)
        finally:
            with sandboxes_lock:
                for sb in active_sandboxes:
                    try:
                        sb.cleanup()
                    except Exception as exc:
                        logger.debug("sandbox cleanup failed: %s", exc)
                active_sandboxes.clear()

        return PipelineResult(
            candidates=len(prs),
            emitted=emitted,
            skipped=sum(skip_reasons.values()),
            out_dir=out_dir,
            skip_reasons=skip_reasons,
        )

    # ----- filters ------------------------------------------------------------

    def _pre_filter(self, pr: PullRequestSummary) -> str | None:
        """Cheap filters that don't need the diff."""
        if pr.is_draft and self.options.skip_drafts:
            return "draft"
        if not pr.merged_at:
            return "not_merged"
        if not pr.changed_files:
            return "no_files"
        if not self.options.prs and _is_non_bug_pr(pr.title):
            # Backports / cherry-picks / release chores / reverts / version
            # bumps aren't real fix tasks, and their bodies leak commit SHAs
            # and fix-PR links. Drop them up front.
            return "non_bug_pr"
        if self.options.require_linked_issue and _linked_issue_number(pr.body or "") is None:
            return "no_linked_issue"
        if (
            self.options.min_problem_statement_words > 0
            and _word_count(pr.body or "") < self.options.min_problem_statement_words
        ):
            return "problem_statement_too_short"
        return None

    def _structural_quality_filter(self, source_patch: str, test_patch: str) -> str | None:
        """Cheap diff-level filters that catch over-emitted task types.

        Returns a skip reason string, or None to keep.

        Two filters here, both shipping a lot of false positives in v0.3:
          1. CI-only PRs: source patch is 100% under .github/. These are
             tooling changes (zizmor scan, publish workflows) — no real
             code-fix signal.
          2. No new test functions: test_patch only edits comments /
             docstrings (e.g. typo cleanup PRs). Without a new test that
             FAILS at base_commit, there can be no FAIL_TO_PASS oracle.
        """
        source_files = _files_in_patch(source_patch)
        # Filter 1: CI-only — all source files are workflow YAMLs
        if (
            self.options.skip_ci_only
            and source_files
            and all(p.startswith(".github/") for p in source_files)
        ):
            return "ci_only_patch"
        # Filter 2: test_patch must add ≥1 new test function
        if not self.options.prs and self.options.require_new_test_funcs:
            n_new = _count_new_test_funcs(test_patch)
            if n_new < 1:
                return "no_new_test_funcs"
        return None

    def _lite_filter(self, pr: PullRequestSummary, source_patch: str) -> str | None:
        """SWE-bench Lite-style structural filters."""
        source_files = _files_in_patch(source_patch)
        if len(source_files) > self.options.max_source_files_per_pr:
            return "too_many_source_files"
        if self.options.lite_filter:
            if len(source_files) != 1:
                return "lite_not_single_source_file"
            if _word_count(pr.body or "") < 40:
                return "lite_problem_too_short"
            # Reject if PR body contains images / external links / cross-PR/issue refs
            body = (pr.body or "").lower()
            if re.search(r"!\[[^\]]*\]\(|<img\s", body):
                return "lite_has_image"
            if re.search(r"\bhttps?://(?!github\.com/)", body):
                return "lite_has_external_link"
            if re.search(r"\b[a-f0-9]{7,40}\b", body):
                return "lite_has_commit_sha"
        return None

    # ----- sandbox -----------------------------------------------------------

    def _start_validation_sandbox(self):
        """Spin up a DockerSandbox from the bootstrap image (shared across PRs)."""
        from repo2rlenv.bootstrap.docker import DockerSandbox

        sandbox = DockerSandbox.start(
            base_image=self.bootstrap.image_tag,
            repo_dir=None,
            platform=self.input.bootstrap.platform,
        )
        check_git = sandbox.exec("test -d /workspace/.git && echo OK", timeout=10)
        if not (check_git.ok and "OK" in check_git.stdout):
            owner, name = self.input.repo.owner_name
            repo_url = f"https://github.com/{owner}/{name}.git"
            clone_res = sandbox.exec(
                f"rm -rf /workspace/.keep && git clone {repo_url} /workspace",
                timeout=300,
            )
            if not clone_res.ok:
                logger.warning("failed to clone %s in sandbox: %s", repo_url, clone_res.stderr)
            check_setup = sandbox.exec("test -f /sandbox/setup.py && echo OK", timeout=10)
            if check_setup.ok and "OK" in check_setup.stdout:
                setup_cmd = f"python3 /sandbox/setup.py --workspace /workspace {owner}/{name}"
                res = sandbox.exec(setup_cmd, timeout=300)
                if not res.ok:
                    logger.warning("setup.py failed: %s", res.stderr)
            elif self.bootstrap.language.value == "python":
                sandbox.exec(
                    "cd /workspace && pip install -e . --no-build-isolation --find-links /wheels",
                    timeout=180,
                )
        return sandbox

    # ----- task builder -------------------------------------------------------

    def _build_task(
        self,
        pr: PullRequestSummary,
        patch: str,
        test_patch: str,
        *,
        fail_to_pass: list[str],
        pass_to_pass: list[str],
        validation_status: str,
    ) -> HarborTask:
        owner, name = self.input.repo.owner_name
        task_id = f"{owner}__{name}-{pr.number}"

        resolved_test_cmds = targeted_test_cmds_for_pr(
            normalize_test_cmds_for_runtime(self.bootstrap.test_cmds),
            _files_in_patch(test_patch),
        )
        eval_script = build_eval_script(
            base_commit=pr.base_sha,
            test_patch=test_patch,
            test_cmds=resolved_test_cmds,
            language=self.bootstrap.language.value,
            fail_to_pass=fail_to_pass,
            pass_to_pass=pass_to_pass,
        )
        image_ref = (
            self.bootstrap.image_digest
            if self.bootstrap.pushed_to_registry
            else self.bootstrap.image_tag
        )

        runtime_metadata = {
            "pr_url": pr.url,
            "pr_merged_at": pr.merged_at,
            "base_commit": pr.base_sha,
            "fail_to_pass": fail_to_pass,
            "pass_to_pass": pass_to_pass,
            "validation_status": validation_status,
            "bootstrap_image": self.bootstrap.image_digest,
            # Graded reward: reward.txt carries f2p_rate*p2p_rate (training
            # signal); the strict SWE-bench `resolved` bool + full breakdown
            # go to reward-details.json (eval signal — Harbor's reward.json
            # schema is flat-numeric-only). See _pr_runtime_verifier.py.
            "reward_mode": "graded",
        }
        repo2env = {
            "pipeline": "pr_swegemma_runtime",
            "pipeline_version": "0.3.0",
            "repo": f"{owner}/{name}",
            "ref": pr.base_sha,
            "reference": pr.url,
            "source_access": self.input.repo.access,
            "built_at": datetime.now(UTC).isoformat(),
            **({"synthesis_llm": self.input.llm.qualified_name} if self.input.llm else {}),
            "reward_kinds": ["test_execution", "diff_similarity"],
            "pr_swegemma_runtime": runtime_metadata,
            "pr_runtime": runtime_metadata,
            # Difficulty + coverage metadata so consumers can slice train/eval
            # by hardness and judge regression-guard strength (p2p_count == 0
            # means no P2P regression guard — weaker eval; see UTBoost).
            "reward_calibration": {
                "f2p_count": len(fail_to_pass),
                "p2p_count": len(pass_to_pass),
                "source_files": len(_files_in_patch(patch)),
                "loc_changed": _diff_loc_changed(patch),
                "difficulty": _difficulty_bucket(len(fail_to_pass), _diff_loc_changed(patch)),
            },
        }

        return HarborTask(
            name=task_id,
            org=self.input.output.org,
            description=pr.title or task_id,
            instruction=_build_instruction(
                pr,
                owner,
                name,
                token=self._token,
                provider=self._provider,
                hard_mode=self.options.hard_mode,
            ),
            oracle_diff=patch,
            repo2env=repo2env,
            difficulty="medium",
            category="bugfix",
            keywords=[name, "pr_swegemma_runtime"],
            docker_image=image_ref,
            environment_dockerfile=None,
            test_script=eval_script,
            # Ship verifier.py + f2p.json + p2p.json as plain, inspectable task
            # artifacts that test.sh reads from /tests (no base64 in test.sh).
            # The egress guard blackholes the hosts that serve this PR's merged
            # diff and checks out base_commit at container startup in entrypoint.
            aux_files={
                **(_runtime_aux_files(fail_to_pass, pass_to_pass) if fail_to_pass else {}),
                "environment/docker-compose.yaml": egress_guard_compose(
                    fix_source_hosts(self.input.repo.source_kind),
                    base_commit=pr.base_sha,
                ),
            },
        )
