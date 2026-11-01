"""Explicit I2-qualified precision policy plus read-only I3 observation."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['ROUTESCALE_I2_PRECISION_MODE'] = 'mixed'
import scripts.launch_i2_precision_service
from serving.i3_observer import install
install()

if __name__ == '__main__':
    from minisgl.server import launch_server
    launch_server()
