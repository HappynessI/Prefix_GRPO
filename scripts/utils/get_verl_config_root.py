#!/usr/bin/env python3
"""获取 verl 包的 trainer/config 目录路径，用于构造 Hydra searchpath。"""
import os
import sys

try:
    import verl
    config_root = os.path.join(os.path.dirname(verl.__file__), "trainer", "config")
    print(config_root)
    sys.exit(0)
except ImportError:
    print("ERROR: verl not installed, cannot determine config path", file=sys.stderr)
    sys.exit(1)
