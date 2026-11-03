import hashlib,json,shutil,subprocess
from pathlib import Path
root=Path.cwd();target=Path('/mnt/d/RouteScale-local-delivery-20261103/build003-header');target.mkdir(exist_ok=False)
assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
base='routescale-sglang:m5-2b4cbfc';cli='/mnt/d/APP/Docker/resources/bin/docker.exe';expected='sha256:4750c580f23ff6c9256f66f7eefac33b2dba09ea52ae27511f7bda00cf282dad'
assert subprocess.check_output([cli,'image','inspect',base,'--format','{{.Id}}'],text=True).strip()==expected
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
old=json.loads((root/'results/delivery/local_v1/sglang_framework_bundle_host_header_attempt.json').read_text())
stock=Path('/usr/local/cuda-13.0/targets/x86_64-linux/include/crt/math_functions.h');shutil.copy2(stock,target/'cuda-math-functions.h')
bundle=dict(old,source_commit=commit,clean_source_checkout=True,source_sha256={n:sha(root/n) for n in old['source_sha256']},container_toolchain_header_sha256=sha(stock),container_toolchain_profile='CUDA13_stock_header_bookworm_glibc2.36',derived_base_image_id=expected,derived_base_source_commit=old['source_commit'],archives_reused_from='build002',bundle_preparer_sha256=sha(root/'scripts/prepare_framework_bundle.py'))
(target/'framework_bundle.json').write_text(json.dumps(bundle,indent=2)+'\n')
p=target/'deploy/local/sglang/Dockerfile.offline';p.parent.mkdir(parents=True);shutil.copy2(root/'deploy/local/sglang/Dockerfile.offline',p)
(target/'Dockerfile').write_text('''FROM routescale-sglang:m5-2b4cbfc
COPY cuda-math-functions.h /opt/cuda/targets/x86_64-linux/include/crt/math_functions.h
COPY deploy/local/sglang/Dockerfile.offline /workspace/RouteScale/deploy/local/sglang/Dockerfile.offline
COPY framework_bundle.json /image_manifest.json
RUN printf '#include <cuda_runtime.h>\\n__global__ void probe(float *x){x[threadIdx.x]=rsqrtf(x[threadIdx.x]);}\\n' > /tmp/header_probe.cu && /opt/cuda/bin/nvcc -std=c++20 -gencode=arch=compute_120f,code=sm_120f -c /tmp/header_probe.cu -o /tmp/header_probe.o && rm /tmp/header_probe.cu /tmp/header_probe.o
''')
print(json.dumps(dict(source_commit=commit,parent_image_id=expected,context=str(target),stock_header_sha256=sha(stock))))
