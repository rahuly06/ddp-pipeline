FROM pytorch/pytorch:2.14.0-cuda12.6-cudnn9-runtime

WORKDIR /app

COPY requirements-docker.txt .
RUN python -m pip install --no-cache-dir -r requirements-docker.txt

COPY train.py .
COPY README.md .

CMD ["python", "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=1", "train.py", "--batch-size", "16", "--epochs", "2"]
