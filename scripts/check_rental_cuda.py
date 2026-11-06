"""Check actual host support in each pinned Torch stack before qualifications."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rental.common import write


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    import torch.distributed as dist
    if not torch.cuda.is_available() or torch.cuda.device_count() != 4 or not dist.is_nccl_available():
        raise ValueError('Four CUDA devices and NCCL required in each environment')
    devices = []
    for index in range(4):
        with torch.cuda.device(index):
            if not torch.cuda.is_bf16_supported():
                raise ValueError('BF16 unsupported')
            value = torch.ones(16, device=f'cuda:{index}')
            if float((value * 2).sum()) != 32:
                raise ValueError('Basic CUDA kernel failed')
            torch.cuda.synchronize()
            devices.append(dict(index=index, name=torch.cuda.get_device_name(index),
                                capability=torch.cuda.get_device_capability(index)))
            del value
            torch.cuda.empty_cache()
    report = dict(passed=True, torch=torch.__version__, cuda=torch.version.cuda,
                  nccl=torch.cuda.nccl.version(), compiled_arches=torch.cuda.get_arch_list(), devices=devices)
    write(args.output, report)
    print(json.dumps(report))
