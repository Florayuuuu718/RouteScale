"""I2 shared-tokenizer server; retain asynchronous execution in lifecycle checks."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "runs/vendor/mini-sglang-i2/python"))

import torch
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from minisgl.engine.sample import Sampler
native_sample = Sampler.sample
from serving.minisgl_observer import install
install()
# I2 has a separate full-vocabulary gate. Do not serialize every forward with
# I1's synchronous finite observation while testing overlap cancellation.
Sampler.sample = native_sample

if __name__ == "__main__":
    from scripts.prepare_i2_minisglang import validate
    validate(ROOT / "runs/vendor/mini-sglang-i2")
    from minisgl.server import launch_server
    launch_server()
