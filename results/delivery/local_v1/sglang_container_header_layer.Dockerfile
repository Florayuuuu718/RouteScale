FROM routescale-sglang:m5-2b4cbfc
COPY cuda-math-functions.h /opt/cuda/targets/x86_64-linux/include/crt/math_functions.h
COPY deploy/local/sglang/Dockerfile.offline /workspace/RouteScale/deploy/local/sglang/Dockerfile.offline
COPY framework_bundle.json /image_manifest.json
RUN printf '#include <cuda_runtime.h>\n__global__ void probe(float *x){x[threadIdx.x]=rsqrtf(x[threadIdx.x]);}\n' > /tmp/header_probe.cu && /opt/cuda/bin/nvcc -std=c++20 -gencode=arch=compute_120f,code=sm_120f -c /tmp/header_probe.cu -o /tmp/header_probe.o && rm /tmp/header_probe.cu /tmp/header_probe.o
