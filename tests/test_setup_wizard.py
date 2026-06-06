"""setup 初始化引导单元测试：.env 合并写入、掩码、完整交互流程（mock input）。

隔离策略：
- MILU_HOME → tmp_path：用户级 .env / config.json 写入临时目录；
- MILU_PROJECT_DIR → tmp_path：避免合并仓库自带的项目级 config/milu.json；
- 引导会直接写 os.environ —— 测试先用 monkeypatch.setenv 预置同名变量，
  teardown 时由 monkeypatch 恢复原值，不污染其他测试。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from milu.cli.setup_wizard import mask_key, run_setup_wizard, update_env_file


# ── 工具函数 ───────────────────────────────────────────────

class TestMaskKey:
    def test_long_key(self):
        assert mask_key("sk-abcdefgh12345678") == "sk-a******5678"

    def test_short_key_fully_masked(self):
        assert mask_key("abc12345") == "********"


class TestUpdateEnvFile:
    def test_create_new_file(self, tmp_path):
        path = tmp_path / "sub" / ".env"
        update_env_file(path, {"QWEN_API_KEY": "sk-test"})
        assert path.read_text(encoding="utf-8") == "QWEN_API_KEY=sk-test\n"

    def test_update_existing_preserves_comments_and_other_keys(self, tmp_path):
        path = tmp_path / ".env"
        path.write_text(
            "# 注释保留\nQWEN_API_KEY=sk-old\nOTHER_KEY=keep\n", encoding="utf-8"
        )
        update_env_file(path, {"QWEN_API_KEY": "sk-new"})
        content = path.read_text(encoding="utf-8")
        assert "# 注释保留" in content
        assert "QWEN_API_KEY=sk-new" in content
        assert "sk-old" not in content
        assert "OTHER_KEY=keep" in content

    def test_append_new_keys(self, tmp_path):
        path = tmp_path / ".env"
        path.write_text("EXISTING=1\n", encoding="utf-8")
        update_env_file(path, {"WEB_SEARCH_PROVIDER": "bocha", "BOCHA_API_KEY": "k1"})
        lines = path.read_text(encoding="utf-8").splitlines()
        assert lines[0] == "EXISTING=1"
        assert "WEB_SEARCH_PROVIDER=bocha" in lines
        assert "BOCHA_API_KEY=k1" in lines

    def test_value_with_space_quoted(self, tmp_path):
        path = tmp_path / ".env"
        update_env_file(path, {"K": "a b"})
        assert 'K="a b"' in path.read_text(encoding="utf-8")


# ── 交互流程（mock input）─────────────────────────────────

@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """把用户数据目录与项目目录都指到 tmp_path，返回 tmp_path。"""
    monkeypatch.setenv("MILU_HOME", str(tmp_path))
    monkeypatch.setenv("MILU_PROJECT_DIR", str(tmp_path))
    return tmp_path


def _feed_inputs(monkeypatch, answers: list[str]) -> None:
    """把预设答案按顺序喂给 input()。答案耗尽后再调用视为测试编排错误。"""
    it = iter(answers)
    monkeypatch.setattr("builtins.input", lambda *a: next(it))


class TestWizardFlow:
    def test_full_flow_writes_env_and_config(self, isolated_home, monkeypatch):
        # 预置同名环境变量：保证 teardown 恢复，且覆盖到「已有 Key 被替换」分支
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-old")
        monkeypatch.setenv("WEB_SEARCH_PROVIDER", "ddg")
        monkeypatch.setenv("BOCHA_API_KEY", "old-bocha")
        _feed_inputs(monkeypatch, [
            "deepseek",        # [1/4] 厂商（按名称选择）
            "",                # [2/4] 模型 → 内置默认
            "sk-new-key-123",  # [3/4] 替换 API Key
            "1",               # [4/4] 搜索后端 → bocha
            "new-bocha-key",   # bocha Key
            "n",               # 不验证
        ])

        assert run_setup_wizard() == 0

        env_content = (isolated_home / ".env").read_text(encoding="utf-8")
        assert "DEEPSEEK_API_KEY=sk-new-key-123" in env_content
        assert "WEB_SEARCH_PROVIDER=bocha" in env_content
        assert "BOCHA_API_KEY=new-bocha-key" in env_content

        config = json.loads((isolated_home / "config.json").read_text(encoding="utf-8"))
        assert config["agent"]["llm"]["provider"] == "deepseek"
        assert config["agent"]["llm"]["model"]  # 内置默认模型已落盘

        # 当前进程即时生效（teardown 由 monkeypatch 恢复）
        assert os.environ["DEEPSEEK_API_KEY"] == "sk-new-key-123"
        assert os.environ["WEB_SEARCH_PROVIDER"] == "bocha"

    def test_keep_existing_key_and_skip_search(self, isolated_home, monkeypatch):
        monkeypatch.setenv("QWEN_API_KEY", "sk-existing")
        _feed_inputs(monkeypatch, [
            "qwen",  # 厂商
            "",      # 模型 → 默认
            "",      # 回车保留现有 Key
            "",      # 搜索后端 → 跳过（默认）
            "",      # 验证 → 默认 N
        ])

        assert run_setup_wizard() == 0

        # 未输入任何新密钥 → 不创建 .env
        assert not (isolated_home / ".env").exists()
        config = json.loads((isolated_home / "config.json").read_text(encoding="utf-8"))
        assert config["agent"]["llm"]["provider"] == "qwen"

    def test_cancel_writes_nothing(self, isolated_home, monkeypatch):
        # EOF（如 Ctrl+Z/Ctrl+D）等价 Ctrl+C：退出且不写任何文件
        def _eof(*_a):
            raise EOFError
        monkeypatch.setattr("builtins.input", _eof)

        assert run_setup_wizard() == 130
        assert not (isolated_home / ".env").exists()
        assert not (isolated_home / "config.json").exists()

    def test_invalid_then_valid_provider_choice(self, isolated_home, monkeypatch):
        monkeypatch.setenv("GLM_API_KEY", "sk-x")
        _feed_inputs(monkeypatch, [
            "nonexistent",  # 无效厂商名 → 重新询问
            "glm",          # 有效
            "", "", "", "",
        ])
        assert run_setup_wizard() == 0
        config = json.loads((isolated_home / "config.json").read_text(encoding="utf-8"))
        assert config["agent"]["llm"]["provider"] == "glm"


# ── CLI 注册 ──────────────────────────────────────────────

def test_parser_registers_setup_subcommand():
    from milu.cli.app import build_parser

    args = build_parser().parse_args(["setup"])
    assert args.command == "setup"


# ── 用户级 .env 兜底加载（milu._env）──────────────────────

def test_user_level_env_fallback(tmp_path, monkeypatch):
    import milu._env as env_mod

    monkeypatch.setenv("MILU_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("MILU_TEST_USER_ENV=hello\n", encoding="utf-8")
    monkeypatch.delenv("MILU_TEST_USER_ENV", raising=False)
    monkeypatch.setattr(env_mod, "_loaded", False)

    env_mod.ensure_dotenv_loaded()
    try:
        assert os.environ.get("MILU_TEST_USER_ENV") == "hello"
    finally:
        os.environ.pop("MILU_TEST_USER_ENV", None)  # dotenv 写入不在 monkeypatch 管辖内


def test_user_level_env_does_not_override_process_env(tmp_path, monkeypatch):
    import milu._env as env_mod

    monkeypatch.setenv("MILU_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("MILU_TEST_PRIORITY=from-file\n", encoding="utf-8")
    monkeypatch.setenv("MILU_TEST_PRIORITY", "from-process")
    monkeypatch.setattr(env_mod, "_loaded", False)

    env_mod.ensure_dotenv_loaded()
    assert os.environ["MILU_TEST_PRIORITY"] == "from-process"
