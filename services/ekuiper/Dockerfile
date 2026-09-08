FROM lfedge/ekuiper:1.13-slim-python

USER root

# Install dependencies required by OpenCV and Python packages
RUN apt-get -o Acquire::Check-Valid-Until=false update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install the Python dependencies for our portable plugin
COPY plugins/ppe_inference/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Revert to the default non-root user for eKuiper
USER kuiper
