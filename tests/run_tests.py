"""运行测试的辅助脚本 - 从 .env 加载环境变量后运行 test_real_api.py"""
import os
import sys
import runpy
from pathlib import Path

# 从 .env 文件加载 API Keys
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())
else:
    print(f"警告: 未找到 {env_path}，请确保 .env 文件存在")

# 运行测试
sys.path.insert(0, os.path.dirname(__file__))
runpy.run_path(os.path.join(os.path.dirname(__file__), "test_real_api.py"), run_name="__main__")
