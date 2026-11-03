FROM routescale-sglang:m5-018cbae
COPY scripts/check_framework_container.py /workspace/RouteScale/scripts/check_framework_container.py
COPY scripts/audit_framework_container.py /workspace/RouteScale/scripts/audit_framework_container.py
COPY docs/sglang_container_protocol.md /workspace/RouteScale/docs/sglang_container_protocol.md
COPY framework_bundle.json /image_manifest.json
