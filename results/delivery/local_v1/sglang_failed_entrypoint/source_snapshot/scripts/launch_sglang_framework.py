"""Native SGLang launcher with one scoped observability repair, also in spawn."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from serving.framework_flush_metrics import install

# Multiprocessing imports this as __mp_main__; each scheduler must install it.
install()

if __name__=='__main__':
    from sglang.launch_server import main
    main()
