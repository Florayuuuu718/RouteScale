"""Read-only SIGUSR1 stack dumps for shutdown diagnosis; never a timing launcher."""
from pathlib import Path
import faulthandler
import signal
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import scripts.launch_i2_precision_service

faulthandler.register(signal.SIGUSR1, all_threads=True)

if __name__=='__main__':
    from minisgl.server import launch_server
    launch_server()
