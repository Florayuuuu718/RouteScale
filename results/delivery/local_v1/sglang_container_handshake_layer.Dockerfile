FROM routescale-sglang:m5-53ba6b2
COPY scripts/warm_framework.py /workspace/RouteScale/scripts/warm_framework.py
COPY scripts/check_framework_container.py /workspace/RouteScale/scripts/check_framework_container.py
COPY scripts/audit_framework_container.py /workspace/RouteScale/scripts/audit_framework_container.py
COPY docs/sglang_container_protocol.md /workspace/RouteScale/docs/sglang_container_protocol.md
COPY serving/framework_gate_client.py /workspace/RouteScale/serving/framework_gate_client.py
COPY framework_bundle.json /image_manifest.json
