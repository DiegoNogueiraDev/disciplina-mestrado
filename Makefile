# Makefile para Pipeline de Análise de Sentimento
# Uso: make <target>

.PHONY: help setup collect preprocess train-baseline train-advanced dashboard clean all test validate

# Configurações
PYTHON = venv/bin/python
TOPIC = "Reforma Tributária"
TWITTER_LIMIT = 800
REDDIT_LIMIT = 200
SAMPLE_SIZE = 600

# Diretórios
DATA_RAW = data/raw
DATA_PROCESSED = data/processed
DATA_OUTPUT = data/output
MODELS_DIR = models
LOGS_DIR = logs
RESULTS_DIR = results
FIGURES_DIR = figures

help:  ## Mostrar esta ajuda
	@echo "📋 Pipeline de Análise de Sentimento - Comandos Disponíveis"
	@echo "=" * 55
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "🚀 Execução típica: make setup && make validate && make all"

# ──── Configuração e Validação ────────────────────────────────
setup:  ## Configurar ambiente e dependências
	@echo "🔧 Configurando ambiente..."
	python -m venv venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	@echo "✅ Ambiente configurado!"

validate:  ## Validar configuração e conectividade
	@echo "🔍 Validando projeto..."
	$(PYTHON) scripts/validate_project.py
	$(PYTHON) scripts/test_reddit_connection.py
	$(PYTHON) scripts/test_fasttext.py
	@echo "✅ Validação completa!"

status:  ## Verificar status dos dados
	@echo "📊 Verificando status dos dados..."
	$(PYTHON) scripts/data_status.py

# ──── Testes ──────────────────────────────────────────────────
test:  ## Executar testes unitários
	@echo "🧪 Executando testes..."
	$(PYTHON) scripts/test_pipeline.py
	@echo "✅ Testes concluídos!"

benchmark:  ## Executar benchmark GPU vs CPU
	@echo "⚡ Executando benchmark de performance..."
	$(PYTHON) scripts/benchmark_performance.py --samples 1000
	@echo "📊 Benchmark salvo em results/benchmark.json"

# ──── Pipeline Principal ──────────────────────────────────────
eda:  ## Executar análise exploratória (EDA)
	@echo "📊 Executando EDA..."
	$(PYTHON) -m jupyter nbconvert --execute notebooks/00_eda.ipynb --to notebook --inplace
	@echo "✅ EDA concluída! Figuras salvas em $(FIGURES_DIR)/"

collect:  ## Coletar dados do Twitter e Reddit
	@echo "📡 Coletando dados..."
	$(PYTHON) scripts/collector.py --limit-twitter $(TWITTER_LIMIT) --limit-reddit $(REDDIT_LIMIT)
	@echo "✅ Coleta concluída! Dados salvos em $(DATA_RAW)/"

preprocess:  ## Pré-processar dados coletados
	@echo "⚙️  Pré-processando dados..."
	$(PYTHON) scripts/preprocess.py --format parquet --hash-users
	@echo "✅ Pré-processamento concluído! Dados salvos em $(DATA_PROCESSED)/"

train:  ## Treinar modelo baseline com métricas
	@echo "🤖 Treinando modelo baseline..."
	$(PYTHON) scripts/train_baseline.py --data-path $(DATA_PROCESSED)/topic.parquet --save-metrics
	@echo "✅ Treinamento concluído! Modelo salvo em $(MODELS_DIR)/"

predict:  ## Executar inferência em lote
	@echo "🔮 Executando predições..."
	$(PYTHON) scripts/predict_batch.py
	@echo "✅ Predições concluídas! Resultados em $(RESULTS_DIR)/"

dashboard:  ## Iniciar dashboard interativo
	@echo "🎨 Iniciando dashboard..."
	$(PYTHON) scripts/dashboard_run.py

# ──── Pipelines Completos ─────────────────────────────────────
all:  ## Executar pipeline completo (coleta → dashboard)
	@echo "🚀 Executando pipeline completo..."
	make collect
	make eda
	make preprocess
	make train
	make predict
	make dashboard

notebooks:  ## Executar todos os notebooks em sequência
	@echo "📓 Executando notebooks..."
	$(PYTHON) -m jupyter nbconvert --execute notebooks/00_eda.ipynb --to notebook --inplace
	$(PYTHON) -m jupyter nbconvert --execute notebooks/01_coleta.ipynb --to notebook --inplace
	$(PYTHON) -m jupyter nbconvert --execute notebooks/02_rotulagem_eda.ipynb --to notebook --inplace
	@echo "✅ Notebooks executados!"

# ──── Limpeza e Manutenção ────────────────────────────────────
clean-data:  ## Limpar apenas dados (manter modelos)
	@echo "🧹 Limpando dados..."
	rm -rf $(DATA_RAW)/* $(DATA_PROCESSED)/* $(DATA_OUTPUT)/*
	@echo "✅ Dados limpos!"

clean-models:  ## Limpar apenas modelos
	@echo "🧹 Limpando modelos..."
	rm -rf $(MODELS_DIR)/*
	@echo "✅ Modelos limpos!"

clean-results:  ## Limpar resultados e figuras
	@echo "🧹 Limpando resultados..."
	rm -rf $(RESULTS_DIR)/* $(FIGURES_DIR)/*
	@echo "✅ Resultados limpos!"

clean:  ## Limpeza completa (dados + modelos + resultados)
	@echo "🧹 Limpeza completa..."
	$(PYTHON) scripts/clean_project.py --all
	@echo "✅ Projeto limpo!"

# ──── Desenvolvimento ─────────────────────────────────────────
install-dev:  ## Instalar dependências de desenvolvimento
	$(PYTHON) -m pip install jupyter lab jupyterlab pytest pytest-cov black flake8
	@echo "✅ Dependências de desenvolvimento instaladas!"

format:  ## Formatar código com black
	$(PYTHON) -m black src/ scripts/
	@echo "✅ Código formatado!"

lint:  ## Verificar qualidade do código
	$(PYTHON) -m flake8 src/ scripts/
	@echo "✅ Linting concluído!"

requirements-lock:  ## Gerar requirements-lock.txt
	$(PYTHON) -m pip freeze > requirements-lock.txt
	@echo "✅ requirements-lock.txt atualizado!"

# ──── Utilitários ─────────────────────────────────────────────
jupyter:  ## Iniciar Jupyter Lab
	@echo "🔬 Iniciando Jupyter Lab..."
	$(PYTHON) -m jupyter lab

logs:  ## Mostrar logs em tempo real
	@echo "📋 Mostrando logs..."
	tail -f $(LOGS_DIR)/sentiment_pipeline.log

gpu-status:  ## Verificar status da GPU
	@echo "🎮 Status da GPU:"
	nvidia-smi

disk-usage:  ## Verificar uso de disco do projeto
	@echo "💾 Uso de disco:"
	du -sh data/ models/ results/ figures/ logs/

# ──── Informações ─────────────────────────────────────────────
info:  ## Mostrar informações do projeto
	@echo "📋 Informações do Projeto"
	@echo "========================="
	@echo "🏷️  Tópico: $(TOPIC)"
	@echo "📊 Limites: Twitter=$(TWITTER_LIMIT), Reddit=$(REDDIT_LIMIT)"
	@echo "🎯 Amostra: $(SAMPLE_SIZE) para rotulagem"
	@echo "🐍 Python: $$($(PYTHON) --version)"
	@echo "📦 Dependências: $$($(PYTHON) -m pip list | wc -l) pacotes"
	@echo ""
	@echo "📁 Estrutura:"
	@echo "   📄 Dados brutos: $$(find $(DATA_RAW) -type f 2>/dev/null | wc -l) arquivos"
	@echo "   ⚙️  Dados processados: $$(find $(DATA_PROCESSED) -type f 2>/dev/null | wc -l) arquivos"
	@echo "   🤖 Modelos: $$(find $(MODELS_DIR) -type f 2>/dev/null | wc -l) arquivos"
	@echo "   📊 Resultados: $$(find $(RESULTS_DIR) -type f 2>/dev/null | wc -l) arquivos"
	@echo "   🎨 Figuras: $$(find $(FIGURES_DIR) -type f 2>/dev/null | wc -l) arquivos"

# ──── Reprodutibilidade ───────────────────────────────────────
reproduce:  ## Reproduzir experimento completo com seed fixa
	@echo "🔄 Reproduzindo experimento..."
	@export PYTHONHASHSEED=42 && make clean && make all
	@echo "✅ Experimento reproduzido!"

# ──── Aliases Úteis ───────────────────────────────────────────
quick-test: validate test benchmark  ## Testes rápidos (validação + unitários + benchmark)

full-pipeline: clean setup validate all  ## Pipeline completo do zero

dev-setup: setup install-dev  ## Configuração para desenvolvimento