import hashlib,json,shutil,subprocess
from pathlib import Path
root=Path.cwd();target=Path('/mnt/d/RouteScale-local-delivery-20261103/build004-warmup');target.mkdir(exist_ok=False)
assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();base='routescale-sglang:m5-428f1ea';expected='sha256:16b41113aed6690b72e8501561a3230b42e6e3aea5dcc303b655785e8ddce3c1';cli='/mnt/d/APP/Docker/resources/bin/docker.exe'
assert subprocess.check_output([cli,'image','inspect',base,'--format','{{.Id}}'],text=True).strip()==expected
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();old=json.loads((root/'results/delivery/local_v1/sglang_framework_bundle_pre_warmup_attempt.json').read_text());names=[*old['source_sha256'],'scripts/warm_framework.py'];assert len(set(names))==92
bundle=dict(old,source_commit=commit,clean_source_checkout=True,source_sha256={n:sha(root/n) for n in names},derived_base_image_id=expected,derived_base_source_commit=old['source_commit'],bundle_preparer_sha256=sha(root/'scripts/prepare_framework_bundle.py'),declared_warmup=dict(requests=24,concurrency=4,output_tokens=[32,64,256],prompt_shapes='shortest/longest eligible I3 prompt',startup_budget_s=600,public_client_budget_s=30))
(target/'framework_bundle.json').write_text(json.dumps(bundle,indent=2)+'\n');p=target/'scripts/warm_framework.py';p.parent.mkdir(parents=True);shutil.copy2(root/'scripts/warm_framework.py',p)
(target/'Dockerfile').write_text('''FROM routescale-sglang:m5-428f1ea
COPY scripts/warm_framework.py /workspace/RouteScale/scripts/warm_framework.py
COPY framework_bundle.json /image_manifest.json
''')
print(json.dumps(dict(source_commit=commit,parent_image_id=expected,context=str(target),source_files=92)))
