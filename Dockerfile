# Container for tyger_recon.py: reads a raw MRD file, runs the GPU k-space
# reconstruction, writes a reconstructed MRD file.
#
#   build: docker build -t xe-tyger-recon .
#   run:   docker run --rm --gpus all -v /path/to/data:/data xe-tyger-recon \
#              --input /data/xe_dyn_raw.mrd --output /data/xe_dyn_recon.mrd
#
# Base image notes:
# - The -devel image is required: the numba CUDA kernels in recon.py need
#   libnvvm, which the -runtime images do not ship.
# - Ubuntu 24.04 provides python3.12, required by the MEDCAP mrd fork.
# - GPU access at runtime needs the NVIDIA container toolkit (--gpus all).
FROM nvidia/cuda:12.6.2-devel-ubuntu24.04

LABEL org.opencontainers.image.source=https://github.com/MEDCAP/asap_recon

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv git \
    && rm -rf /var/lib/apt/lists/*

# venv sidesteps Ubuntu 24.04's externally-managed-environment pip restriction
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY gtypes.py raw.py recon.py results.py tyger_recon.py ./

ENTRYPOINT ["python", "tyger_recon.py"]
