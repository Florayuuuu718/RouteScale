FROM lmsysorg/sglang@sha256:b27fce60bc5494c118c4910702812bcfa8cee67abcdd1ff8b0902f21647552f4
WORKDIR /workspace/RouteScale
COPY serving/ ./serving/
COPY scripts/framework_service.py ./scripts/framework_service.py
COPY scripts/check_framework.py ./scripts/check_framework.py
COPY scripts/audit_framework.py ./scripts/audit_framework.py
COPY deploy/local/sglang/ ./deploy/local/sglang/
COPY config/delivery/model.json ./config/delivery/model.json
COPY results/i_minisglang/i3_online_v2/workload002/workload.json ./results/i_minisglang/i3_online_v2/workload002/workload.json
COPY results/i_minisglang/i3_online_v2/online001/schedule.json ./results/i_minisglang/i3_online_v2/online001/schedule.json
COPY docs/sglang_delivery_protocol.md ./docs/sglang_delivery_protocol.md
ENV ROUTESCALE_IN_CONTAINER=1 ROUTESCALE_FRAMEWORK_PYTHON=python3 ROUTESCALE_MODEL_PATH=/models/Qwen3-0.6B
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
EXPOSE 1971 1972
HEALTHCHECK --start-period=600s --interval=10s --timeout=3s CMD python3 -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:1971/ready',timeout=2)"
ENTRYPOINT ["sh", "deploy/local/sglang/entrypoint.sh"]
