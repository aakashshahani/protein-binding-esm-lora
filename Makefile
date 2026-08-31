# Convenience targets. On Windows, run the underlying python commands directly.
.PHONY: install install-gpu lint test data cluster embed train-bilstm train-frozen eval serve docker

install:
	pip install -r requirements.txt && pip install -e .

install-gpu:
	pip install -r requirements-gpu.txt --index-url https://download.pytorch.org/whl/cu124
	pip install -r requirements.txt && pip install -e .

lint:
	ruff check .

test:
	pytest -q

data:
	python scripts/download_data.py --config configs/data.yaml
	python scripts/build_dataset.py --config configs/data.yaml

cluster:
	python scripts/cluster_split.py --config configs/data.yaml

embed:
	python scripts/extract_embeddings.py --config configs/bilstm.yaml --splits train valid test

train-bilstm:
	python scripts/train.py --config configs/bilstm.yaml

train-frozen:
	python scripts/train.py --config configs/frozen_head.yaml

train-lora-local:
	python scripts/train.py --config configs/lora.yaml --track local

eval:
	python scripts/evaluate.py --run outputs/bilstm_esm2_t33_650M_UR50D

serve:
	uvicorn pbsite.serve.api:app --reload --port 8000

docker:
	docker build -f docker/Dockerfile -t pbsite-serve .
