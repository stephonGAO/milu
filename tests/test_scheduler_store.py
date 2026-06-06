"""ScheduleStore 多用户存储测试：分文件隔离、迁移、原子写。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from milu.scheduler.store import ScheduleStore, ScheduleTask, _safe_user


def _mk(name: str, user: str = "default", trigger: str = "once") -> ScheduleTask:
    return ScheduleTask.create(
        name=name,
        prompt="测试 prompt",
        trigger_type=trigger,
        run_at="2099-01-01T09:00:00",
        user_id=user,
    )


class TestSafeUser:
    def test_normal(self):
        assert _safe_user("alice") == "alice"

    def test_sanitized(self):
        assert _safe_user("a/b:c") == "a_b_c"

    def test_empty_falls_back_default(self):
        assert _safe_user("") == "default"
        assert _safe_user("   ") == "default"

    def test_truncated_to_64(self):
        assert len(_safe_user("x" * 100)) == 64


class TestMultiUserStore:
    def test_per_user_file_isolation(self, tmp_path: Path):
        store = ScheduleStore(tmp_path)
        store.add(_mk("t1", "alice"))
        store.add(_mk("t2", "bob"))
        assert (tmp_path / "schedules" / "alice.json").exists()
        assert (tmp_path / "schedules" / "bob.json").exists()
        assert [t.name for t in store.list_user("alice")] == ["t1"]
        assert [t.name for t in store.list_user("bob")] == ["t2"]

    def test_same_name_different_user_ok(self, tmp_path: Path):
        """任务名唯一性是用户内约束，跨用户同名可共存。"""
        store = ScheduleStore(tmp_path)
        store.add(_mk("daily", "alice"))
        store.add(_mk("daily", "bob"))  # 不冲突
        with pytest.raises(ValueError):
            store.add(_mk("daily", "alice"))  # 同用户重名才冲突

    def test_get_remove_scoped_by_user(self, tmp_path: Path):
        store = ScheduleStore(tmp_path)
        store.add(_mk("t", "alice"))
        assert store.get("t", "alice") is not None
        assert store.get("t", "bob") is None      # 跨用户不可见
        assert store.remove("t", "bob") is False  # 跨用户不可删
        assert store.remove("t", "alice") is True

    def test_update_scoped_by_user(self, tmp_path: Path):
        store = ScheduleStore(tmp_path)
        store.add(_mk("t", "alice"))
        task = store.get("t", "alice")
        task.run_count = 7
        assert store.update(task) is True
        assert store.get("t", "alice").run_count == 7

    def test_list_all_spans_users(self, tmp_path: Path):
        store = ScheduleStore(tmp_path)
        store.add(_mk("t1", "alice"))
        store.add(_mk("t2", "bob"))
        store.add(_mk("t3", "default"))
        names = {t.name for t in store.list_all()}
        assert names == {"t1", "t2", "t3"}

    def test_sanitized_user_path_roundtrip(self, tmp_path: Path):
        """非法字符 user_id 落到安全化文件名，读写一致。"""
        store = ScheduleStore(tmp_path)
        store.add(_mk("t", "a/b:c"))
        assert (tmp_path / "schedules" / "a_b_c.json").exists()
        assert store.get("t", "a/b:c").name == "t"

    def test_no_tmp_residue_after_save(self, tmp_path: Path):
        """原子写完成后不残留 .tmp 临时文件。"""
        store = ScheduleStore(tmp_path)
        store.add(_mk("t", "alice"))
        assert list((tmp_path / "schedules").glob("*.tmp")) == []


class TestBackwardCompat:
    def test_old_json_without_user_id(self, tmp_path: Path):
        """旧格式任务（无 user_id 字段）反序列化自动补 default。"""
        sched_dir = tmp_path / "schedules"
        sched_dir.mkdir(parents=True)
        (sched_dir / "default.json").write_text(
            json.dumps({"tasks": [{
                "id": "1", "name": "old", "prompt": "p", "trigger_type": "once",
                "run_at": "2099-01-01T09:00:00",
            }]}),
            encoding="utf-8",
        )
        store = ScheduleStore(tmp_path)
        tasks = store.list_user("default")
        assert len(tasks) == 1
        assert tasks[0].user_id == "default"

    def test_legacy_migration(self, tmp_path: Path):
        """旧单文件 schedules.json 首次访问迁移为 schedules/default.json。"""
        legacy = tmp_path / "schedules.json"
        legacy.write_text(
            json.dumps({"tasks": [{
                "id": "1", "name": "old", "prompt": "p", "trigger_type": "once",
                "run_at": "2099-01-01T09:00:00",
            }]}),
            encoding="utf-8",
        )
        store = ScheduleStore(tmp_path)
        tasks = store.list_user("default")
        assert [t.name for t in tasks] == ["old"]
        # 源文件保留为 .migrated（可回退），不删除
        assert not legacy.exists()
        assert (tmp_path / "schedules.json.migrated").exists()

    def test_legacy_migration_idempotent(self, tmp_path: Path):
        """target 已存在时不重复迁移（不覆盖已有多用户数据）。"""
        sched_dir = tmp_path / "schedules"
        sched_dir.mkdir(parents=True)
        (sched_dir / "default.json").write_text(
            json.dumps({"tasks": []}), encoding="utf-8"
        )
        legacy = tmp_path / "schedules.json"
        legacy.write_text(json.dumps({"tasks": [{"id": "1", "name": "x",
                          "prompt": "p", "trigger_type": "once"}]}), encoding="utf-8")
        ScheduleStore(tmp_path)
        # legacy 保留原位（未迁移未改名），default.json 未被覆盖
        assert legacy.exists()
        assert ScheduleStore(tmp_path).list_user("default") == []

    def test_corrupt_legacy_not_migrated(self, tmp_path: Path):
        """损坏的旧文件不迁移不删除，留给人工处理。"""
        legacy = tmp_path / "schedules.json"
        legacy.write_text("{broken json", encoding="utf-8")
        store = ScheduleStore(tmp_path)
        assert legacy.exists()
        assert store.list_user("default") == []
