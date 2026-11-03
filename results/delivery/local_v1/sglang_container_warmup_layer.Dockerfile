FROM routescale-sglang:m5-428f1ea
COPY scripts/warm_framework.py /workspace/RouteScale/scripts/warm_framework.py
COPY framework_bundle.json /image_manifest.json
