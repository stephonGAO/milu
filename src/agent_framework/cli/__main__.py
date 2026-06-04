"""支持 `python -m agent_framework.cli`。"""
import sys

from agent_framework.cli.app import main

if __name__ == "__main__":
    sys.exit(main())
