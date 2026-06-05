"""测试 .env 加载助手（去除库对 CWD 的隐式依赖）。"""
import dotenv

import milu._env as envmod


def test_ensure_dotenv_loaded_is_idempotent(monkeypatch):
    """ensure_dotenv_loaded 进程内最多触发一次 load_dotenv。"""
    monkeypatch.setattr(envmod, "_loaded", False, raising=False)
    monkeypatch.delenv("MILU_NO_DOTENV", raising=False)

    calls = {"n": 0}
    monkeypatch.setattr(dotenv, "load_dotenv",
                        lambda *a, **k: calls.__setitem__("n", calls["n"] + 1))

    envmod.ensure_dotenv_loaded()
    envmod.ensure_dotenv_loaded()
    envmod.ensure_dotenv_loaded()

    assert calls["n"] == 1, "应仅加载一次"


def test_ensure_dotenv_disabled_by_env(monkeypatch):
    """设置 MILU_NO_DOTENV 后不应扫描 CWD。"""
    monkeypatch.setattr(envmod, "_loaded", False, raising=False)
    monkeypatch.setenv("MILU_NO_DOTENV", "1")

    calls = {"n": 0}
    monkeypatch.setattr(dotenv, "load_dotenv",
                        lambda *a, **k: calls.__setitem__("n", calls["n"] + 1))

    envmod.ensure_dotenv_loaded()

    assert calls["n"] == 0, "关闭后不应调用 load_dotenv"


def test_ensure_dotenv_never_raises(monkeypatch):
    """load_dotenv 抛异常时也不应让库崩溃。"""
    monkeypatch.setattr(envmod, "_loaded", False, raising=False)
    monkeypatch.delenv("MILU_NO_DOTENV", raising=False)

    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(dotenv, "load_dotenv", boom)
    envmod.ensure_dotenv_loaded()  # 不抛异常即通过


def test_load_env_exported_from_package():
    """load_env 应作为公开 API 从顶层导出。"""
    import milu
    assert hasattr(milu, "load_env")
    assert "load_env" in milu.__all__


def test_disabled_truthy_variants(monkeypatch):
    for val in ("1", "true", "TRUE", "Yes", "on"):
        monkeypatch.setenv("MILU_NO_DOTENV", val)
        assert envmod._disabled() is True, f"{val} 应视为关闭"
    for val in ("0", "false", "", "no"):
        monkeypatch.setenv("MILU_NO_DOTENV", val)
        assert envmod._disabled() is False, f"{val} 不应视为关闭"
