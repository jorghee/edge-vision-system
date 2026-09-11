FROM lfedge/ekuiper:1.13-slim-python

USER root

# libglib2.0-0 is needed by opencv-python-headless at runtime.
# libgl1 is NOT needed because we use the headless variant of OpenCV.
RUN sed -i 's/deb.debian.org/archive.debian.org/g' /etc/apt/sources.list && \
    sed -i 's|security.debian.org/debian-security|archive.debian.org/debian-security|g' /etc/apt/sources.list || true && \
    sed -i '/bullseye-updates/d' /etc/apt/sources.list && \
    apt-get -o Acquire::Check-Valid-Until=false update && \
    apt-get install -y --fix-missing libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Install the Python dependencies for our portable plugin
COPY plugins/ppe_inference/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Revert to the default non-root user for eKuiper
