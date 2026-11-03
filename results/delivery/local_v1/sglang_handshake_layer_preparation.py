import hashlib,json,shutil,subprocess
from pathlib import Path
root=Path.cwd();target=Path('/mnt/d/RouteScale-local-delivery-20261103/build005-handshake');target.mkdir(exist_ok=False)
assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();base='routescale-sglang:m5-53ba6b2';expected='sha256:a7d5701eac1495040a98aa5f43f9d3d56e7312f6f9043c45d476b2e56c2a75ea';cli='/mnt/d/APP/Docker/resources/bin/docker.exe'
assert subprocess.check_output([cli,'image','inspect',base,'--format','{{.Id}}'],text=True).strip()==expected
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();old=json.loads((root/'results/delivery/local_v1/sglang_framework_bundle_fixed_window_attempt.json').read_text());added=['scripts/check_framework_container.py','scripts/audit_framework_container.py','docs/sglang_container_protocol.md','serving/framework_gate_client.py'];names=[*old['source_sha256'],*added];assert len(set(names))==96
bundle=dict(old,source_commit=commit,clean_source_checkout=True,source_sha256={n:sha(root/n) for n in names},derived_base_image_id=expected,derived_base_source_commit=old['source_commit'],bundle_preparer_sha256=sha(root/'scripts/prepare_framework_bundle.py'),declared_warmup=dict(requests=28,concurrency=4,output_tokens=[32,64,256,900],prompt_shapes='shortest/longest eligible I3; 900 only shortest',startup_budget_s=600,public_client_budget_s=30),diagnostic_scope='container_queue_handshake_v1')
(target/'framework_bundle.json').write_text(json.dumps(bundle,indent=2)+'\n');files=['scripts/warm_framework.py',*added]
for name in files:
 p=target/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/name,p)
(target/'Dockerfile').write_text('FROM '+base+'\n'+''.join('COPY '+n+' /workspace/RouteScale/'+n+'\n' for n in files)+'COPY framework_bundle.json /image_manifest.json\n')
print(json.dumps(dict(source_commit=commit,parent_image_id=expected,context=str(target),source_files=len(names))))
