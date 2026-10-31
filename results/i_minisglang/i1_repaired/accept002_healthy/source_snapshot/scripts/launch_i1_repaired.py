"""Launch the replayed I1 sources; observation contains no runtime repairs."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
CHECKOUT = Path(os.environ.get("ROUTESCALE_MINISGL_CHECKOUT", ROOT / "runs/vendor/mini-sglang-i1-fixed"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CHECKOUT / "python"))

from serving.minisgl_observer import install

install()

if __name__ == "__main__":
    from scripts.prepare_i1_lifecycle import validate
    validate(CHECKOUT)
    from minisgl.server import launch_server
    launch_server()
