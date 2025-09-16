.PHONY: lint build run download gen test

IMAGE ?= ghcr.io/herugen/index-tts2:latest

build:
	docker build -f Dockerfile -t $(IMAGE) .

download:
	docker run --rm -v $(PWD):/local python:3.10-slim bash -lc "pip install modelscope && modelscope download --model IndexTeam/IndexTTS-2 --local_dir /local/checkpoints"

run: download
	docker run -d --rm -p 9010:9010 \
		-v ./huggingface:/root/.cache/huggingface \
		-v ./checkpoints:/app/checkpoints \
		-e HF_ENDPOINT="https://hf-mirror.com" \
		--name indextts-http \
		$(IMAGE)

logs:
	docker logs -f indextts-http

stop:
	docker stop indextts-http

lint:
	docker run --rm -v $(PWD):/redocly redocly/cli:latest lint /redocly/openapi.yml --format=stylish

test:
	python test.py --examples-dir examples --outputs-dir outputs
