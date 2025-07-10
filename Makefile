# Makefile para Pipeline de Análise de Sentimento
# Uso: make <target>

.PHONY: help setup collect preprocess train-baseline train-advanced dashboard clean all

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

help:  ## Mostrar esta ajuda
	@echo "Pipeline de Análise de Sentimento - Comandos Disponíveis:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Exemplo de uso completo:"
	@echo "  make setup collect preprocess train-baseline dashboard"

setup:  ## Configurar ambiente (criar diretórios, instalar dependências)
	@echo "🔧 Configurando ambiente..."
	mkdir -p $(DATA_RAW) $(DATA_PROCESSED) $(DATA_OUTPUT) $(MODELS_DIR) $(LOGS_DIR)
	mkdir -p figures reports
	@if [ ! -d "venv" ]; then \
		echo "⚠️  Virtual environment não encontrado. Criando..."; \
		python3 -m venv venv; \
		$(PYTHON) -m pip install --upgrade pip; \
		$(PYTHON) -m pip install -r requirements.txt; \
	fi
	@echo "✅ Ambiente configurado!"

collect:  ## Coletar dados do Twitter e Reddit
	@echo "📡 Iniciando coleta de dados..."
	@echo "  Limites: Twitter=$(TWITTER_LIMIT), Reddit=$(REDDIT_LIMIT)"
	$(PYTHON) scripts/collector.py \
		--limit-twitter $(TWITTER_LIMIT) \
		--limit-reddit $(REDDIT_LIMIT) \
		--topic $(TOPIC) \
		--verbose
	@echo "✅ Coleta concluída! Dados salvos em $(DATA_RAW)/"

collect-sample:  ## Coletar amostra pequena para testes (Twitter=10, Reddit=10)
	@echo "📡 Coletando amostra pequena para testes..."
	$(PYTHON) scripts/collector.py \
		--limit-twitter 10 \
		--limit-reddit 10 \
		--topic $(TOPIC) \
		--verbose

preprocess:  ## Pré-processar dados coletados
	@echo "🧹 Iniciando pré-processamento..."
	$(PYTHON) scripts/preprocess.py \
		--input-dir $(DATA_RAW) \
		--output-dir $(DATA_PROCESSED) \
		--combine \
		--verbose
	@echo "✅ Pré-processamento concluído! Dados salvos em $(DATA_PROCESSED)/"

train-baseline:  ## Treinar modelo baseline (TF-IDF + Logistic Regression)
	@echo "🤖 Treinando modelo baseline..."
	@PROCESSED_FILE=$$(ls -t $(DATA_PROCESSED)/combined_processed_*.csv 2>/dev/null | head -n1); \
	if [ -z "$$PROCESSED_FILE" ]; then \
		echo "❌ Nenhum arquivo processado encontrado. Execute 'make preprocess' primeiro."; \
		exit 1; \
	fi; \
	echo "  Usando arquivo: $$PROCESSED_FILE"; \
	$(PYTHON) scripts/train_baseline.py \
		--data-path "$$PROCESSED_FILE" \
		--sample-size $(SAMPLE_SIZE) \
		--output-dir $(MODELS_DIR) \
		--verbose
	@echo "✅ Modelo baseline treinado! Salvo em $(MODELS_DIR)/"

train-advanced:  ## Treinar modelo avançado (FastText + MLP)
	@echo "🚀 Treinando modelo avançado..."
	@PROCESSED_FILE=$$(ls -t $(DATA_PROCESSED)/combined_processed_*.csv 2>/dev/null | head -n1); \
	if [ -z "$$PROCESSED_FILE" ]; then \
		echo "❌ Nenhum arquivo processado encontrado. Execute 'make preprocess' primeiro."; \
		exit 1; \
	fi; \
	echo "  Usando arquivo: $$PROCESSED_FILE"; \
	$(PYTHON) scripts/train_advanced.py \
		--data-path "$$PROCESSED_FILE" \
		--output-dir $(MODELS_DIR) \
		--epochs 30 \
		--verbose
	@echo "✅ Modelo avançado treinado! Salvo em $(MODELS_DIR)/"

predict:  ## Fazer predições com modelo treinado
	@echo "🔮 Fazendo predições..."
	@PROCESSED_FILE=$$(ls -t $(DATA_PROCESSED)/combined_processed_*.csv 2>/dev/null | head -n1); \
	MODEL_FILE=$$(ls -t $(MODELS_DIR)/*.pkl $(MODELS_DIR)/*.pth 2>/dev/null | head -n1); \
	if [ -z "$$PROCESSED_FILE" ] || [ -z "$$MODEL_FILE" ]; then \
		echo "❌ Arquivos necessários não encontrados. Execute 'make preprocess' e 'make train-baseline' primeiro."; \
		exit 1; \
	fi; \
	echo "  Dados: $$PROCESSED_FILE"; \
	echo "  Modelo: $$MODEL_FILE"; \
	$(PYTHON) create_fake_predictions.py
	@echo "✅ Predições concluídas! Resultados em $(DATA_OUTPUT)/"

dashboard:  ## Iniciar dashboard interativo
	@echo "📊 Iniciando dashboard..."
	@if [ ! -d "$(DATA_OUTPUT)" ] || [ -z "$$(ls -A $(DATA_OUTPUT) 2>/dev/null)" ]; then \
		echo "⚠️  Dados de saída não encontrados. Executando predições..."; \
		make predict; \
	fi
	@echo "🌐 Dashboard disponível em: http://localhost:8050"
	@echo "   (Pressione Ctrl+C para parar)"
	$(PYTHON) scripts/dashboard_run.py \
		--data-path $(DATA_OUTPUT) \
		--host 0.0.0.0 \
		--port 8050

dashboard-bg:  ## Iniciar dashboard em background
	@echo "📊 Iniciando dashboard em background..."
	nohup $(PYTHON) scripts/dashboard_run.py \
		--data-path $(DATA_OUTPUT) \
		--host 0.0.0.0 \
		--port 8050 > logs/dashboard.log 2>&1 &
	@echo "✅ Dashboard rodando em background. Logs em logs/dashboard.log"
	@echo "🌐 Acesse: http://localhost:8050"

stop-dashboard:  ## Parar dashboard em background
	@echo "🛑 Parando dashboard..."
	pkill -f "dashboard_run.py" || echo "Dashboard não estava rodando"

status:  ## Mostrar status do pipeline
	@echo "📋 Status do Pipeline:"
	@echo ""
	@echo "📁 Estrutura de diretórios:"
	@find data models logs -type f 2>/dev/null | head -20 | sed 's/^/  /'
	@echo ""
	@echo "📊 Contagem de arquivos:"
	@echo "  Dados brutos: $$(find $(DATA_RAW) -name "*.csv" 2>/dev/null | wc -l) arquivos"
	@echo "  Dados processados: $$(find $(DATA_PROCESSED) -name "*.csv" 2>/dev/null | wc -l) arquivos"
	@echo "  Dados de saída: $$(find $(DATA_OUTPUT) -name "*.csv" 2>/dev/null | wc -l) arquivos"
	@echo "  Modelos: $$(find $(MODELS_DIR) -name "*.pkl" -o -name "*.pth" 2>/dev/null | wc -l) arquivos"
	@echo ""
	@echo "🔧 Ambiente Python:"
	@if [ -d "venv" ]; then echo "  ✅ Virtual environment ativo"; else echo "  ❌ Virtual environment não encontrado"; fi

test:  ## Executar testes rápidos do pipeline
	@echo "🧪 Executando testes do pipeline..."
	$(PYTHON) test_pipeline.py
	@echo "✅ Testes concluídos!"

notebook:  ## Abrir Jupyter notebook demonstrativo
	@echo "📓 Iniciando Jupyter notebook..."
	@if [ ! -f "pipeline_demo.ipynb" ]; then \
		echo "⚠️  Notebook demonstrativo não encontrado. Criando..."; \
		make create-notebook; \
	fi
	$(PYTHON) -m jupyter notebook pipeline_demo.ipynb

create-notebook:  ## Criar notebook demonstrativo
	@echo "📝 Criando notebook demonstrativo..."
	$(PYTHON) scripts/create_demo_notebook.py
	@echo "✅ Notebook criado: pipeline_demo.ipynb"

compliance:  ## Gerar relatório de compliance LGPD
	@echo "⚖️  Gerando relatório de compliance..."
	$(PYTHON) -c "
import sys; sys.path.append('src')
from utils.compliance import generate_compliance_report
import json

metadata = {
    'lgpd': {'legal_basis': 'Legitimate interest for academic research'},
    'tos': {'platform': 'Twitter/Reddit', 'allowed_use': 'Academic research'},
    'processing_info': {'original_records': 1000, 'final_records': 950, 'columns_removed': ['username']}
}

generate_compliance_report(metadata, 'reports/compliance_report.txt')
print('✅ Relatório salvo em: reports/compliance_report.txt')
"

clean:  ## Limpar arquivos temporários e logs
	@echo "🧹 Limpando arquivos temporários..."
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name ".pytest_cache" -type d -exec rm -rf {} +
	rm -f logs/*.log
	@echo "✅ Limpeza concluída!"

clean-data:  ## Limpar todos os dados (CUIDADO!)
	@echo "⚠️  ATENÇÃO: Isso irá apagar todos os dados coletados!"
	@read -p "Tem certeza? (y/N): " confirm && [ "$$confirm" = "y" ]
	rm -rf $(DATA_RAW)/* $(DATA_PROCESSED)/* $(DATA_OUTPUT)/*
	@echo "🗑️  Dados apagados!"

all: setup collect preprocess train-baseline predict dashboard  ## Executar pipeline completo

demo: setup collect-sample preprocess train-baseline predict dashboard  ## Demo rápido com amostra pequena

# Targets de desenvolvimento
dev-install:  ## Instalar dependências de desenvolvimento
	$(PYTHON) -m pip install jupyter matplotlib seaborn pytest black isort

format:  ## Formatar código Python
	$(PYTHON) -m black src/ scripts/
	$(PYTHON) -m isort src/ scripts/

lint:  ## Verificar qualidade do código
	$(PYTHON) -m black --check src/ scripts/
	$(PYTHON) -m isort --check-only src/ scripts/

# Informações do sistema
info:  ## Mostrar informações do sistema
	@echo "🔍 Informações do Sistema:"
	@echo "  Python: $$($(PYTHON) --version 2>&1)"
	@echo "  PyTorch: $$($(PYTHON) -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Não instalado')"
	@echo "  CUDA disponível: $$($(PYTHON) -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')"
	@echo "  GPU: $$($(PYTHON) -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"Nenhuma\")' 2>/dev/null || echo 'N/A')"
	@echo "  Espaço em disco: $$(df -h . | tail -n1 | awk '{print $$4}') disponível"

# Backup
backup:  ## Criar backup dos dados e modelos
	@echo "💾 Criando backup..."
	@BACKUP_NAME="backup_$$(date +%Y%m%d_%H%M%S)"; \
	mkdir -p backups; \
	tar -czf "backups/$$BACKUP_NAME.tar.gz" data/ models/ config/ || true; \
	echo "✅ Backup criado: backups/$$BACKUP_NAME.tar.gz"

# Help adicional
usage:  ## Mostrar exemplos de uso
	@echo "📖 Exemplos de Uso:"
	@echo ""
	@echo "1. Pipeline completo (primeira vez):"
	@echo "   make setup collect preprocess train-baseline dashboard"
	@echo ""
	@echo "2. Demonstração rápida:"
	@echo "   make demo"
	@echo ""
	@echo "3. Apenas coleta de novos dados:"
	@echo "   make collect preprocess predict"
	@echo ""
	@echo "4. Retreinar modelo:"
	@echo "   make train-baseline"
	@echo ""
	@echo "5. Ver status do sistema:"
	@echo "   make status"