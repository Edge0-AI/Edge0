FROM nvidia/cuda@sha256:5dc1bca23d05bd37b011be68ec470c03b403a5da07ec3a86e41af9470e9d0cc6
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends build-essential cmake ninja-build git ca-certificates curl python3 python3-venv python3-dev pkg-config && rm -rf /var/lib/apt/lists/*
ENV HOME=/workspace/home PYTHONUNBUFFERED=1
WORKDIR /workspace/Strata
