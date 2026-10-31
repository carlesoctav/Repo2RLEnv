"""Unit tests for DockerSandbox and docker helper functions."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock

import pytest

from repo2rlenv.bootstrap.docker import DockerError, DockerSandbox, ExecResult, _run


def test_docker_sandbox_exec_pipes_command_via_stdin(monkeypatch):
    sandbox = DockerSandbox(container_id="cid-12345", repo_mount="", platform="linux/amd64")
    recorded = {}

    def fake_run(args, *, timeout=600, input_text=None):
        recorded["args"] = args
        recorded["timeout"] = timeout
        recorded["input_text"] = input_text
        return ExecResult(exit_code=0, stdout="output", stderr="", duration_sec=0.1)

    monkeypatch.setattr("repo2rlenv.bootstrap.docker._run", fake_run)

    res = sandbox.exec("echo hello\npytest", timeout=45)
    assert res.ok
    assert recorded["args"] == ["docker", "exec", "-w", "/workspace", "-i", "cid-12345", "bash", "-l"]
    assert recorded["timeout"] == 45
    assert recorded["input_text"] == "echo hello\npytest"


def test_docker_sandbox_exec_raises_if_not_alive():
    sandbox = DockerSandbox(container_id="cid-12345", repo_mount="", platform="linux/amd64")
    sandbox._alive = False

    with pytest.raises(DockerError, match="sandbox has been cleaned up; cannot exec"):
        sandbox.exec("echo hello")


def test_docker_run_passes_input_text(monkeypatch):
    recorded = {}

    def fake_subprocess_run(args, **kwargs):
        recorded["args"] = args
        recorded["input"] = kwargs.get("input")
        mock = MagicMock()
        mock.returncode = 0
        mock.stdout = "done"
        mock.stderr = ""
        return mock

    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)

    res = _run(["docker", "exec", "-i", "cid", "bash", "-l"], timeout=100, input_text="test command")
    assert res.ok
    assert recorded["args"] == ["docker", "exec", "-i", "cid", "bash", "-l"]
    assert recorded["input"] == "test command"


def test_docker_sandbox_commit_sets_workdir(monkeypatch):
    sandbox = DockerSandbox(container_id="cid-12345", repo_mount="", platform="linux/amd64", workdir="/custom/workdir")
    recorded = []

    def fake_run(args, *, timeout=600, input_text=None):
        recorded.append(args)
        if "inspect" in args:
            return ExecResult(exit_code=0, stdout='[{"Id": "sha256:abc"}]', stderr="", duration_sec=0.1)
        return ExecResult(exit_code=0, stdout="digest", stderr="", duration_sec=0.1)

    monkeypatch.setattr("repo2rlenv.bootstrap.docker._run", fake_run)
    sandbox.commit("test-tag")
    commit_args = recorded[0]
    assert commit_args == [
        "docker",
        "commit",
        "-c",
        "WORKDIR /custom/workdir",
        "cid-12345",
        "test-tag",
    ]

