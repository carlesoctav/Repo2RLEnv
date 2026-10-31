"""Unit tests for the bootstrap agent parser."""

from __future__ import annotations

from repo2rlenv.bootstrap.agent import parse_action


def test_parse_action_with_thought():
    text = "Thought: inspect files\nAction: BASH\nInput: ls -la"
    thought, action = parse_action(text)
    assert thought == "inspect files"
    assert action.name == "BASH"
    assert action.input == "ls -la"


def test_parse_action_without_thought():
    text = "Action: BASH\nInput: pytest"
    thought, action = parse_action(text)
    assert thought == ""
    assert action.name == "BASH"
    assert action.input == "pytest"


def test_parse_action_whitespace_leniency():
    text = "Action:   READ_FILE   Input:   pyproject.toml"
    thought, action = parse_action(text)
    assert thought == ""
    assert action.name == "READ_FILE"
    assert action.input == "pyproject.toml"


def test_parse_action_with_thought_and_multiple_spaces():
    text = "Thought:  need to run setup  \n\n  Action:  SAVE_SETUP  \n  Input:  {\"rebuild_cmds\": [], \"test_cmds\": []}"
    thought, action = parse_action(text)
    assert thought == "need to run setup"
    assert action.name == "SAVE_SETUP"
    assert action.input == "{\"rebuild_cmds\": [], \"test_cmds\": []}"


def test_do_complete_translates_native_tool_call(monkeypatch):
    import litellm
    from repo2rlenv.llm import _do_complete
    from repo2rlenv.spec.input import LLMSpec

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    spec = LLMSpec(provider="openai", model="thinkingmachines/Inkling", reasoning_effort="high")

    class ToolCallFunction:
        name = "BASH"
        arguments = '{"input": "pip list"}'

    class ToolCall:
        function = ToolCallFunction()

    class Message:
        content = ""
        tool_calls = [ToolCall()]

    class Choice:
        message = Message()

    class DummyResponse:
        choices = [Choice()]
        usage = {"prompt_tokens": 10, "completion_tokens": 20}

    monkeypatch.setattr(litellm, "completion", lambda **kwargs: DummyResponse())
    monkeypatch.setattr(litellm, "completion_cost", lambda **kwargs: 0.0)

    resp = _do_complete(spec, system=None, user="hi", max_tokens=1000, temperature=0.0)
    assert resp.content == "Thought: Execute BASH\nAction: BASH\nInput: pip list"
    thought, action = parse_action(resp.content)
    assert thought == "Execute BASH"
    assert action.name == "BASH"
    assert action.input == "pip list"


def test_do_complete_translates_save_setup_tool_call(monkeypatch):
    import litellm
    from repo2rlenv.llm import _do_complete
    from repo2rlenv.spec.input import LLMSpec

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    spec = LLMSpec(provider="openai", model="thinkingmachines/Inkling", reasoning_effort="high")

    class ToolCallFunction:
        name = "SAVE_SETUP"
        arguments = '{"rebuild_cmds": ["pip install -e ."], "test_cmds": ["pytest"]}'

    class ToolCall:
        function = ToolCallFunction()

    class Message:
        content = None
        tool_calls = [ToolCall()]

    class Choice:
        message = Message()

    class DummyResponse:
        choices = [Choice()]
        usage = None

    monkeypatch.setattr(litellm, "completion", lambda **kwargs: DummyResponse())
    monkeypatch.setattr(litellm, "completion_cost", lambda **kwargs: 0.0)

    resp = _do_complete(spec, system=None, user="hi", max_tokens=1000, temperature=0.0)
    assert "Action: SAVE_SETUP" in resp.content
    thought, action = parse_action(resp.content)
    assert action.name == "SAVE_SETUP"
    assert '"rebuild_cmds": ["pip install -e ."]' in action.input

