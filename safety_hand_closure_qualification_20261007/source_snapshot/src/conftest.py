# 让 pytest 在任意 cwd 下都能 import safeduo（src 加进 sys.path）
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
