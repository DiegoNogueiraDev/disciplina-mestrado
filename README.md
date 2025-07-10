# 🎯 Pipeline de Monitoramento de Sentimento - Redes Sociais PT-BR

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-green.svg)](https://developer.nvidia.com/cuda-downloads)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Projeto de Pesquisa**: Pipeline automatizado para coleta, processamento e análise de sentimento em posts sobre tópicos específicos no X/Twitter e Reddit, com foco em conteúdo em português brasileiro.

## 📋 Visão Geral

Este projeto implementa um pipeline completo de análise de sentimento para redes sociais, desde a coleta de dados até a visualização de resultados. Desenvolvido com foco em **experimentos científicos** e **reprodutibilidade**, o sistema permite:

- **Coleta automatizada** de dados do X/Twitter (via snscrape) e Reddit (via PRAW)
- **Pré-processamento** especializado para português brasileiro com SpaCy
- **Rotulagem assistida** usando Label Studio
- **Modelos de ML** otimizados para GPU (FastText + MLP)
- **Dashboard interativo** para visualização de resultados

## 🏗️ Arquitetura

```
disciplina-mestrado/
├── 📊 notebooks/           # Jupyter notebooks para experimentos
│   ├── 01_coleta.ipynb            # Coleta e validação de dados
│   ├── 02_rotulagem_eda.ipynb     # Análise exploratória e rotulagem
│   └── pipeline_demo.ipynb        # Demo completo do pipeline
├── 🔧 src/                 # Código-fonte modular
│   ├── scrapers/           # Coletores de dados
│   ├── preprocessing/      # Limpeza e normalização
│   ├── models/            # Modelos de ML
│   ├── dashboard/         # Interface web
│   └── utils/            # Utilitários
├── 📁 data/               # Dados do projeto
│   ├── raw/              # Dados brutos coletados
│   ├── processed/        # Dados processados
│   └── labeled/          # Dados rotulados
├── 🚀 scripts/            # Scripts de automação
│   ├── collector.py      # Coleta de dados
│   ├── preprocess.py     # Pré-processamento
│   ├── train_baseline.py # Treinamento de modelos
│   └── dashboard_run.py  # Dashboard
├── ⚙️ config/             # Configurações
│   └── topic.yaml        # Definição de tópicos
└── 📚 docs/               # Documentação
```

## 🚀 Início Rápido

### 1. Configuração do Ambiente

```bash
# Clone o repositório
git clone <repository-url>
cd disciplina-mestrado

# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Instalar dependências
pip install -r requirements.txt
```

### 2. Configuração de Credenciais

```bash
# Copiar template de configuração
cp .env.example .env

# Editar credenciais do Reddit
nano .env
```

**Variáveis necessárias no `.env`:**
```bash
# Reddit API (obrigatório)
REDDIT_ID=your_client_id
REDDIT_SECRET=your_client_secret
REDDIT_AGENT=your_user_agent
REDDIT_USERNAME=your_username
REDDIT_PASSWORD=your_password
```

### 3. Configuração do Tópico

Edite `config/topic.yaml` para definir seu tópico de interesse:

```yaml
topic: "Reforma Tributária"
keywords:
  - "reforma tributária"
  - "imposto"
  - "tributação"
limits:
  twitter: 800
  reddit: 200
```

### 4. Execução via Notebooks

**Abordagem recomendada** para experimentos e desenvolvimento:

```bash
# Iniciar Jupyter Lab
jupyter lab

# Executar notebooks na ordem:
# 1. notebooks/01_coleta.ipynb
# 2. notebooks/02_rotulagem_eda.ipynb
# 3. notebooks/pipeline_demo.ipynb (demo completo)
```

### 5. Execução via Scripts

**Abordagem para produção** e automação:

```bash
# 1. Coletar dados
python scripts/collector.py --limit-twitter 800 --limit-reddit 200

# 2. Pré-processar
python scripts/preprocess.py

# 3. Rotular dados (Label Studio)
label-studio --port 8080

# 4. Treinar modelo
python scripts/train_baseline.py

# 5. Executar inferência
python scripts/predict_batch.py

# 6. Visualizar resultados
python scripts/dashboard_run.py
```

## 📊 Notebooks Disponíveis

### 🔍 01_coleta.ipynb
- **Objetivo**: Validação e execução da coleta de dados
- **Conteúdo**: 
  - Teste de conectividade com APIs
  - Coleta de amostras pequenas para validação
  - Execução de coleta completa
  - Verificação de qualidade dos dados

### 🏷️ 02_rotulagem_eda.ipynb
- **Objetivo**: Análise exploratória e preparação para rotulagem
- **Conteúdo**:
  - Estatísticas descritivas dos dados
  - Análise temporal e por plataforma
  - Preparação de amostras para rotulagem
  - Configuração do Label Studio

### 🎯 pipeline_demo.ipynb
- **Objetivo**: Demonstração completa do pipeline
- **Conteúdo**:
  - Execução end-to-end do pipeline
  - Análise de resultados
  - Visualizações e métricas
  - Comparação entre plataformas

## 🛠️ Funcionalidades Principais

### 📡 Coleta de Dados
- **X/Twitter**: Scraping via `snscrape` (sem API key)
- **Reddit**: API oficial via `praw`
- **Filtros**: Idioma, localização, qualidade
- **Formato**: CSV padronizado com timestamp

### 🔧 Pré-processamento
- **SpaCy PT-BR**: Tokenização, lematização, POS tagging
- **Limpeza**: URLs, menções, hashtags, emojis
- **Normalização**: Lowercase, remoção de caracteres especiais
- **Filtros**: Comprimento mínimo/máximo, detecção de idioma

### 🏷️ Rotulagem
- **Label Studio**: Interface web para rotulagem
- **Classes**: Positivo, Negativo, Neutro
- **Métricas**: Concordância inter-anotador
- **Exportação**: JSON/CSV para treinamento

### 🤖 Modelos
- **FastText**: Embeddings contextuais
- **MLP**: Rede neural para classificação
- **GPU**: Treinamento acelerado com CUDA
- **Avaliação**: Métricas completas (F1, Precision, Recall)

### 📊 Dashboard
- **Dash/Plotly**: Interface interativa
- **Visualizações**: Séries temporais, distribuições
- **Filtros**: Data, plataforma, sentimento
- **Métricas**: Volume, engagement, tendências

## 🎯 Casos de Uso

### Pesquisa Acadêmica
- Análise de opinião pública sobre políticas
- Monitoramento de tendências sociais
- Comparação entre plataformas digitais

### Monitoramento Empresarial
- Análise de sentimento sobre produtos/serviços
- Monitoramento de crises de reputação
- Inteligência competitiva

### Análise Social
- Estudos de comportamento digital
- Análise de polarização política
- Monitoramento de eventos em tempo real

## 📐 Metodologia Científica

### Reprodutibilidade
- **Versionamento**: Git para código, DVC para dados
- **Ambientes**: Docker containers e requirements fixos
- **Sementes**: Random seeds fixas para ML
- **Documentação**: Notebooks com análises detalhadas

### Validação
- **Divisão**: Train/validation/test estratificada
- **Métricas**: Balanced accuracy, F1-macro, AUC
- **Baseline**: Comparação com modelos simples
- **Ablation**: Análise de contribuição dos componentes

### Ética e Privacidade
- **Dados públicos**: Apenas conteúdo público
- **Anonimização**: Hash de identificadores
- **LGPD**: Compliance com legislação brasileira
- **Transparência**: Metodologia documentada

## 🔧 Configuração Avançada

### GPU Setup (CUDA 12.1)
```bash
# Verificar CUDA
nvidia-smi

# Instalar PyTorch com CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Tuning de Performance
```yaml
# config/performance.yaml
batch_size: 32
num_workers: 4
pin_memory: true
mixed_precision: true
```

### Monitoramento
```bash
# Logs em tempo real
tail -f logs/sentiment_pipeline.log

# Métricas do sistema
htop
nvidia-smi -l 1
```

## 📊 Resultados Esperados

### Datasets
- **Volume**: 1000+ posts por execução
- **Qualidade**: >95% em português brasileiro
- **Distribuição**: Balanceada entre plataformas
- **Cobertura**: Múltiplos aspectos do tópico

### Modelos
- **Acurácia**: >80% em teste
- **F1-Score**: >0.75 macro
- **Latência**: <100ms por predição
- **Throughput**: 1000+ posts/minuto

### Visualizações
- **Tendências**: Séries temporais por plataforma
- **Distribuições**: Sentimentos por tópico
- **Comparações**: X/Twitter vs Reddit
- **Insights**: Padrões e anomalias

## 🧪 Testes e Validação

### Testes Unitários
```bash
# Executar testes
python -m pytest tests/

# Cobertura
python -m pytest --cov=src tests/
```

### Testes de Integração
```bash
# Testar pipeline completo
python scripts/test_pipeline.py

# Validar dados
python scripts/validate_data.py
```

### Testes de Performance
```bash
# Benchmark de coleta
python scripts/benchmark_collection.py

# Benchmark de inferência
python scripts/benchmark_inference.py
```

## ✅ Status do Projeto

### Validação Completa
```bash
# Execute para verificar se tudo está funcionando
python scripts/validate_project.py

# Resultado esperado: 🎉 PROJETO VÁLIDO!
```

### Funcionalidades Testadas
- ✅ **Estrutura**: Todos os diretórios e arquivos essenciais
- ✅ **Dependências**: Todas as bibliotecas instaladas corretamente
- ✅ **CUDA**: GPU RTX 3060 Ti detectada e funcional
- ✅ **Scrapers**: TwitterScraper e RedditScraper funcionais
- ✅ **Notebooks**: Implementados e prontos para uso
- ✅ **Scripts**: Todos executáveis e funcionais
- ✅ **Fallback**: Sistema Selenium funciona quando snscrape falha

### Limitações Conhecidas
- **snscrape**: Pode falhar com SSL em alguns ambientes (fallback Selenium ativo)
- **Reddit**: Requer credenciais válidas no arquivo `.env`
- **X/Twitter**: Rate limits podem afetar coletas grandes

## 📊 Estrutura de Dados

### Dados Coletados
```
data/
├── raw/                    # Dados brutos por execução
│   └── YYYYMMDD_HHMMSS/   # Timestamp da coleta
│       ├── twitter_data.csv
│       └── reddit_data.csv
├── processed/              # Dados limpos e normalizados
└── labeled/               # Dados rotulados para treinamento
```

### Formatos de Saída
```csv
# Twitter
timestamp,text,user_hash,platform,retweet_count,favorite_count
2025-07-10 00:30:00,"Texto do tweet",abc123,twitter,5,20

# Reddit  
timestamp,title,selftext,subreddit,score,platform,user_hash
2025-07-10 00:30:00,"Título","Texto do post",brasil,15,reddit,def456
```

## 📚 Documentação Adicional

- **[Definição do Projeto](docs/definition.md)**: Escopo detalhado e objetivos
- **[Implementação MVP](docs/implementacao-mvp.md)**: Versão mínima viável
- **[API Reference](docs/api.md)**: Documentação dos módulos
- **[Troubleshooting](docs/troubleshooting.md)**: Soluções para problemas comuns

## 🤝 Contribuição

### Desenvolvimento
1. Fork o repositório
2. Crie uma branch para sua feature
3. Implemente testes
4. Submeta um pull request

### Reportar Issues
- Use templates de issue
- Inclua logs relevantes
- Descreva passos para reproduzir

## 📄 Licença

Este projeto está licenciado sob a MIT License - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 🙏 Agradecimentos

- **SpaCy**: Processamento de linguagem natural
- **PRAW**: Reddit API wrapper
- **snscrape**: Twitter scraping
- **Label Studio**: Plataforma de rotulagem
- **Dash**: Framework para dashboards

---

**Mantido por**: Diego Nogueira  
**Contato**: devnogueiradiego@gmail.com 
**Última atualização**: Julho 2025
