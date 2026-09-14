PYTHON := python
IMAGE := telco-churn
PORT := 8888

.PHONY: help install preprocess eda train tune notebook docker-build docker-run clean all

help:
	@echo "Comandos disponíveis:"
	@echo "  make install       - instala as dependências"
	@echo "  make preprocess    - executa o pré-processamento"
	@echo "  make eda           - executa o script de EDA"
	@echo "  make train         - treina o modelo"
	@echo "  make tune          - executa otimização de hiperparâmetros"
	@echo "  make notebook      - inicia o Jupyter Lab"
	@echo "  make docker-build  - cria a imagem Docker"
	@echo "  make docker-run    - executa o container"
	@echo "  make clean         - remove caches Python"
	@echo "  make all           - preprocess + train"

install:
	$(PYTHON) -m pip install -r requirements.txt

preprocess:
	$(PYTHON) scripts/data_preprocessed.py

eda:
	$(PYTHON) scripts/eda.py

train:
	$(PYTHON) src/models/train.py

tune:
	$(PYTHON) src/models/tune_model.py

notebook:
	jupyter lab

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm -it -p $(PORT):8888 -v "$(CURDIR):/app" $(IMAGE)

clean:
	$(PYTHON) -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"

all: preprocess train
