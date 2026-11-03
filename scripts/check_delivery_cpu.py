"""Lightweight CI for authored delivery files; no model, CUDA or Torch import."""
import ast
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]


def check():
    files=[p for base in ('serving','scripts','tests/delivery') for p in (ROOT/base).glob('*.py')]
    for p in files:ast.parse(p.read_text(),filename=str(p))
    configs=[ROOT/n for n in ('config/delivery/local.json','config/delivery/tp_matrix.json',
        'config/delivery/model.json','deploy/local/sglang/runtime.json','deploy/local/sglang/compose.json','deploy/local/sglang/compose.init.json','deploy/local/sglang/image.lock.json','deploy/local/sglang/upstream_sources.json')]
    for p in configs:json.loads(p.read_text())
    docs=[ROOT/n for n in ('docs/delivery_protocol.md','docs/serving_delivery.md','docs/sglang_delivery_protocol.md',
        'docs/sglang_serving_delivery.md','docs/sglang_cancel_sourcewalk.md','docs/sglang_container_protocol.md','results/delivery/local_v1/README.md')]
    links=0
    for p in docs:
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            target=target.strip('<>').split('#',1)[0]
            if not target or '://' in target:continue
            if not (p.parent/target).exists():raise ValueError(f'Broken link: {p.relative_to(ROOT)} -> {target}')
            links+=1
    print(json.dumps(dict(passed=True,python_AST=len(files),JSON_configs=len(configs),local_doc_links=links,requires_GPU=False)))


if __name__=='__main__':check()
