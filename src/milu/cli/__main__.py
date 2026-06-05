"""支持 `python -m milu.cli`。"""
import sys

from milu.cli.app import main

if __name__ == "__main__":
    sys.exit(main())
