"""Dispatch PR/issue data calls to the right host client by source kind.

`github.py` and `gitlab.py` expose the same surface (`list_merged_prs`,
`fetch_pr_summary`, `fetch_pr_diff`, `fetch_issue`, returning `PullRequestSummary`). Pipelines
call `provider_for(repo).list_merged_prs(...)` instead of importing a host
module directly, so the same mining code works on GitHub and GitLab.
"""

from __future__ import annotations

from types import ModuleType

from repo2rlenv import github, gitlab
from repo2rlenv.github import PullRequestSummary
from repo2rlenv.sources import SourceKind
from repo2rlenv.spec.input import RepoSpec


def provider_for(repo: RepoSpec) -> ModuleType:
    """Return the host client module for this repo's source."""
    if repo.source_kind == SourceKind.GITLAB:
        return gitlab
    return github


def fetch_pr_summary(
    repo: RepoSpec,
    owner: str,
    name: str,
    number: int,
    *,
    token: str | None = None,
) -> PullRequestSummary | None:
    """Dispatch fetch_pr_summary to the host client for repo."""
    return provider_for(repo).fetch_pr_summary(owner, name, number, token=token)
