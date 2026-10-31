"""Same-input first-norm intervention, separate from full-model gates and timing."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    directory = args.directory
    output = directory / 'norm_probe.json'
    if output.exists():
        raise ValueError('Preserve same-input probe')
    lock = (ROOT / 'runs/tmp/h2_runner.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    import torch
    from safetensors import safe_open
    from flashinfer import rmsnorm
    from serving.i2_workloads import digest
    from serving.i2_external_alignment import math_norm
    from scripts.audit_i2_external_alignment import difference
    torch.set_num_threads(4)
    meta = json.loads((directory / 'provenance.json').read_text())
    traces = {}
    for mode in ('native_bf16', 'hf_bf16', 'hf_fp32'):
        artifact = json.loads((directory / f'{mode}.json').read_text())['tensors']['trace']
        if digest(ROOT / artifact['path']) != artifact['sha256']:
            raise ValueError('Trace tensor changed')
        traces[mode] = torch.load(ROOT / artifact['path'], mmap=True, weights_only=True)
    x = traces['native_bf16']['embedding'].cuda()
    if not torch.equal(x.cpu(), traces['hf_bf16']['embedding']) or not torch.equal(x.cpu().float(), traces['hf_fp32']['embedding']):
        raise ValueError('Probe must have identical actual operator inputs')
    weights = ROOT / meta['config']['model_path']
    weight = None
    quantization = dict(elements=0, changed_elements=0, nonfinite_elements=0, max_abs=0.)
    for file in weights.glob('*.safetensors'):
        if digest(file) != meta['model_sha256'][file.name]:
            raise ValueError('Checkpoint changed')
        with safe_open(file, framework='pt', device='cpu') as handle:
            for name in handle.keys():
                value = handle.get_tensor(name)
                if name == 'model.layers.0.input_layernorm.weight':
                    weight = value.cuda()
                original = value.float()
                converted = value.half().float()
                quantization['elements'] += value.numel()
                quantization['changed_elements'] += int((original != converted).sum())
                quantization['nonfinite_elements'] += int((~torch.isfinite(converted)).sum())
                quantization['max_abs'] = max(quantization['max_abs'], float((original-converted).abs().max()))
    if weight is None:
        raise ValueError('Missing norm checkpoint weight')
    with torch.inference_mode():
        native = rmsnorm(x, weight, 1e-6)
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        hf_bf16 = normalized.bfloat16() * weight
        fused_math = math_norm(x, weight, 1e-6)
        fp32 = normalized * weight.float()
    tensors = {name: value.cpu() for name, value in dict(input=x, weight=weight, flashinfer=native,
        hf_bf16=hf_bf16, fused_math=fused_math, fp32=fp32).items()}
    validation = ROOT / meta['validation_dir'] / 'norm_probe.pt'
    torch.save(tensors, validation)
    comparisons = dict(
        flashinfer_reproduces_native_trace=difference(tensors['flashinfer'], traces['native_bf16']['0.input_layernorm.0']),
        hf_bf16_reproduces_hf_trace=difference(tensors['hf_bf16'], traces['hf_bf16']['0.input_layernorm.0']),
        fp32_reproduces_hf_trace=difference(tensors['fp32'], traces['hf_fp32']['0.input_layernorm.0']),
        native_vs_hf_bf16=difference(tensors['flashinfer'], tensors['hf_bf16']),
        native_vs_fp32=difference(tensors['flashinfer'], tensors['fp32']),
        native_vs_single_rounding_math=difference(tensors['flashinfer'], tensors['fused_math']))
    checks = dict(same_input=True,
        native_reproduced=comparisons['flashinfer_reproduces_native_trace']['exact'],
        hf_bf16_reproduced=comparisons['hf_bf16_reproduces_hf_trace']['exact'],
        hf_fp32_reproduced=comparisons['fp32_reproduces_hf_trace']['exact'])
    snapshot = directory / 'norm_probe_source_snapshot'
    snapshot.mkdir()
    sources = ['scripts/probe_i2_external_norm.py', 'scripts/audit_i2_external_alignment.py', 'serving/i2_external_alignment.py']
    for name in sources:
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    record = dict(evidence_class='same_input_first_norm_diagnostic', performance_allowed=False,
        checks=checks, passed=all(checks.values()), comparisons=comparisons,
        bf16_checkpoint_to_fp16=quantization, tensor=dict(path=str(validation.relative_to(ROOT)), sha256=digest(validation)),
        source_sha256={n: digest(ROOT/n) for n in sources},
        scope='first cold prefill last128 rows, layer0 input norm only; not a full-model attribution')
    output.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))
    if not record['passed']:
        raise ValueError('Probe failed to reproduce observed operator outputs')


if __name__ == '__main__':
    main()
