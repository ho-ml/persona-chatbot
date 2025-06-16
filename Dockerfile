FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

WORKDIR /workspace
COPY . .

# environment variable
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# system package
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y -qq \
    git \
    wget \
    python3 \
    python3-pip \
    curl \
    build-essential \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# pytorch package
RUN python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# jupyter notebook
RUN python3 -m pip install jupyterlab notebook ipywidgets

# port
EXPOSE 8888

# jupyter notebook
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]