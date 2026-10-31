"""Launch pinned Mini-SGLang with observation hooks also installed in spawn children."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serving.minisgl_observer import install

install()

if __name__ == "__main__":
    from minisgl.server import launch_server
    launch_server()
