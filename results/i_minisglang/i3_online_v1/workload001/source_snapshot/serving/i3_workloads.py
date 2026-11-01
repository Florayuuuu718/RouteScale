"""Frozen text payloads and genuinely open arrival schedules for I3."""
import copy
import json
from pathlib import Path
import random

from serving.i2_workloads import digest, token_hash

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'config/serving/i3_online.json'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def make_workload(config):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ROOT/config['model_path'], local_files_only=True)
    passage = ('A reliable service records each request, schedules bounded work, and releases its memory. '
               'A cache reuses a common prefix while each question retains its own identity. ')
    body_ids = tokenizer.encode(passage * 100, add_special_tokens=False)
    shared = tokenizer.decode(body_ids[:384])
    rows = []
    kinds = ['short', 'short', 'long', 'shared'] * (config['requests']//4)
    for i, kind in enumerate(kinds):
        size = 80 if kind == 'short' else 440
        message = (shared + f'\nQuestion {i}: Explain one cache tradeoff in a numbered list.' if kind == 'shared'
            else f'Request {i}: ' + tokenizer.decode(body_ids[:size]) + '\nExplain this briefly.')
        prompt = tokenizer.apply_chat_template([dict(role='user', content=message)], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer.encode(prompt)
        budget = dict(short=16, long=32, shared=64)[kind]
        payload = dict(model=str(ROOT/config['model_path']), prompt=prompt, max_tokens=budget,
            ignore_eos=True, temperature=0, top_k=1, top_p=1)
        rows.append(dict(request_id=f'load_{i:03}', kind=kind, payload=payload,
            payload_sha256=token_hash(payload), input_ids=ids, prompt_tokens=len(ids), output_budget=budget))
    random.Random(config['seed']).shuffle(rows)
    warmup = []
    for i, row in enumerate([next(r for r in rows if r['kind'] == k) for k in ('short','long','shared','short')]):
        r = copy.deepcopy(row)
        r['request_id'] = f'warmup_{i}'
        r['payload']['prompt'] = r['payload']['prompt'].replace('Explain', f'Warmup {i}: explain', 1)
        r['input_ids'] = tokenizer.encode(r['payload']['prompt'])
        r['prompt_tokens'] = len(r['input_ids'])
        r['payload_sha256'] = token_hash(r['payload'])
        warmup.append(r)
    if not all(r['prompt_tokens']+r['output_budget'] < config['engine']['max_seq_len'] for r in rows+warmup):
        raise ValueError('Workload exceeds context budget')
    return dict(seed=config['seed'], requests=rows, warmup=warmup,
        tokenization='pre-rendered Qwen chat template, thinking disabled; HTTP tokenizer is measured',
        generation='native greedy; ignore_eos with fixed per-class max_tokens; no forced continuation')


def schedules(workload, capacities, config):
    low, high = min(capacities.values()), max(capacities.values())
    rule = config['rate_rule']
    rates = dict(low=low*rule['low_min_capacity_multiplier'], near=low*rule['near_min_capacity_multiplier'],
        overload=high*rule['overload_max_capacity_multiplier'])
    # One dimensionless exponential trace, reused for every rate and both policies.
    rng = random.Random(config['seed']+17)
    gaps = [rng.expovariate(1) for _ in range(len(workload['requests'])-1)]
    scale = len(gaps)/sum(gaps)
    offsets = [0.]
    for gap in gaps:
        offsets.append(offsets[-1]+gap*scale)
    return dict(capacities_rps=capacities, rates_rps=rates,
        interpretation='near anchors the slower policy; overload exceeds faster pilot capacity; finite trace, not steady state',
        traces={name:[dict(request_id=r['request_id'], arrival_s=t/rate) for r,t in zip(workload['requests'],offsets,strict=True)]
                for name,rate in rates.items()}, latency_objectives_s=config['latency_objectives_s'])
