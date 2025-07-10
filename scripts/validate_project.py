#!/usr/bin/env python3
"""
Script de validação completa do projeto
Verifica se todos os componentes estão funcionando corretamente
"""

import sys
import os
import subprocess
import importlib
from pathlib import Path
import yaml
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class ProjectValidator:
    """Validador completo do projeto"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.errors = []
        self.warnings = []
        
    def log_error(self, message):
        """Registrar erro"""
        self.errors.append(message)
        logger.error(f"❌ {message}")
        
    def log_warning(self, message):
        """Registrar aviso"""
        self.warnings.append(message)
        logger.warning(f"⚠️  {message}")
        
    def log_success(self, message):
        """Registrar sucesso"""
        logger.info(f"✅ {message}")
        
    def validate_project_structure(self):
        """Validar estrutura do projeto"""
        logger.info("🔍 Validando estrutura do projeto...")
        
        required_dirs = [
            "src", "src/scrapers", "src/utils", "src/models", 
            "src/dashboard", "notebooks", "scripts", "config", 
            "data", "data/raw", "data/processed"
        ]
        
        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            if full_path.exists():
                self.log_success(f"Diretório existe: {dir_path}")
            else:
                self.log_error(f"Diretório ausente: {dir_path}")
                # Criar diretório se não existir
                full_path.mkdir(parents=True, exist_ok=True)
                self.log_success(f"Diretório criado: {dir_path}")
        
        required_files = [
            "requirements.txt", "config/topic.yaml", ".env.example",
            "src/scrapers/__init__.py", "src/utils/__init__.py",
            "scripts/collector.py", "notebooks/01_coleta.ipynb"
        ]
        
        for file_path in required_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                self.log_success(f"Arquivo existe: {file_path}")
            else:
                self.log_error(f"Arquivo ausente: {file_path}")
    
    def validate_dependencies(self):
        """Validar dependências Python"""
        logger.info("🐍 Validando dependências Python...")
        
        required_packages = [
            "pandas", "numpy", "matplotlib", "seaborn",
            "praw", "spacy", "fasttext-wheel",
            "torch", "transformers", "dash", "plotly",
            "yaml", "python-dotenv", "langdetect"
        ]
        
        for package in required_packages:
            try:
                # Ajustar nomes de pacotes para importação
                import_name = package.replace('-', '_')
                if package == "fasttext-wheel":
                    import_name = "fasttext"
                elif package == "python-dotenv":
                    import_name = "dotenv"
                elif package == "yaml":
                    import_name = "yaml"
                
                importlib.import_module(import_name)
                self.log_success(f"Pacote disponível: {package}")
            except ImportError:
                self.log_error(f"Pacote ausente: {package}")
    
    def validate_config(self):
        """Validar arquivos de configuração"""
        logger.info("⚙️  Validando configurações...")
        
        # Validar topic.yaml
        config_file = self.project_root / "config" / "topic.yaml"
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                
                required_keys = ['topic', 'keywords', 'limits', 'reddit']
                for key in required_keys:
                    if key in config:
                        self.log_success(f"Configuração válida: {key}")
                    else:
                        self.log_error(f"Chave ausente em topic.yaml: {key}")
                        
            except Exception as e:
                self.log_error(f"Erro ao ler topic.yaml: {e}")
        else:
            self.log_error("Arquivo topic.yaml não encontrado")
        
        # Validar .env.example
        env_example = self.project_root / ".env.example"
        if env_example.exists():
            self.log_success("Arquivo .env.example existe")
        else:
            self.log_error("Arquivo .env.example não encontrado")
    
    def validate_scrapers(self):
        """Validar módulos de scraping"""
        logger.info("🔧 Validando scrapers...")
        
        # Adicionar src ao path
        sys.path.insert(0, str(self.project_root / "src"))
        
        try:
            from scrapers.reddit import RedditScraper
            self.log_success("RedditScraper importado com sucesso")
        except ImportError as e:
            self.log_error(f"Erro ao importar RedditScraper: {e}")
        
        try:
            from utils.config import load_config
            self.log_success("Utilitários de configuração importados")
        except ImportError as e:
            self.log_error(f"Erro ao importar utilitários: {e}")
    
    def validate_notebooks(self):
        """Validar notebooks Jupyter"""
        logger.info("📓 Validando notebooks...")
        
        notebooks = [
            "notebooks/01_coleta.ipynb",
            "notebooks/02_rotulagem_eda.ipynb"
        ]
        
        for notebook in notebooks:
            nb_path = self.project_root / notebook
            if nb_path.exists():
                self.log_success(f"Notebook existe: {notebook}")
                # Verificar se tem conteúdo
                if nb_path.stat().st_size > 1000:  # >1KB
                    self.log_success(f"Notebook tem conteúdo: {notebook}")
                else:
                    self.log_warning(f"Notebook parece vazio: {notebook}")
            else:
                self.log_error(f"Notebook ausente: {notebook}")
    
    def validate_scripts(self):
        """Validar scripts executáveis"""
        logger.info("🚀 Validando scripts...")
        
        scripts = [
            "scripts/collector.py",
            "scripts/preprocess.py", 
            "scripts/train_baseline.py",
            "scripts/predict_batch.py",
            "scripts/dashboard_run.py"
        ]
        
        for script in scripts:
            script_path = self.project_root / script
            if script_path.exists():
                self.log_success(f"Script existe: {script}")
                # Verificar se é executável
                if os.access(script_path, os.X_OK):
                    self.log_success(f"Script é executável: {script}")
                else:
                    self.log_warning(f"Script não é executável: {script}")
            else:
                self.log_error(f"Script ausente: {script}")
    
    def validate_environment(self):
        """Validar ambiente e variáveis"""
        logger.info("🌍 Validando ambiente...")
        
        # Verificar Python version
        python_version = sys.version_info
        if python_version >= (3, 8):
            self.log_success(f"Python versão OK: {python_version.major}.{python_version.minor}")
        else:
            self.log_error(f"Python versão inadequada: {python_version.major}.{python_version.minor} (mínimo 3.8)")
        
        # Verificar CUDA se disponível
        try:
            import torch
            if torch.cuda.is_available():
                self.log_success(f"CUDA disponível: {torch.cuda.get_device_name()}")
            else:
                self.log_warning("CUDA não disponível (usando CPU)")
        except ImportError:
            self.log_warning("PyTorch não instalado")
        
        # Verificar espaço em disco
        disk_usage = os.statvfs(self.project_root)
        free_space_gb = (disk_usage.f_frsize * disk_usage.f_bavail) / (1024**3)
        if free_space_gb > 5:
            self.log_success(f"Espaço em disco OK: {free_space_gb:.1f} GB livres")
        else:
            self.log_warning(f"Pouco espaço em disco: {free_space_gb:.1f} GB livres")
    
    def run_validation(self):
        """Executar validação completa"""
        logger.info("🔍 INICIANDO VALIDAÇÃO COMPLETA DO PROJETO")
        logger.info("="*60)
        
        validation_steps = [
            self.validate_project_structure,
            self.validate_dependencies,
            self.validate_config,
            self.validate_scrapers,
            self.validate_notebooks,
            self.validate_scripts,
            self.validate_environment
        ]
        
        for step in validation_steps:
            try:
                step()
            except Exception as e:
                self.log_error(f"Erro na validação {step.__name__}: {e}")
            logger.info("-" * 40)
        
        # Relatório final
        self.print_summary()
    
    def print_summary(self):
        """Imprimir resumo da validação"""
        logger.info("📊 RESUMO DA VALIDAÇÃO")
        logger.info("="*60)
        
        total_checks = len(self.errors) + len(self.warnings)
        
        if not self.errors and not self.warnings:
            logger.info("🎉 PROJETO VÁLIDO! Todos os componentes estão funcionando.")
        elif not self.errors:
            logger.info(f"✅ PROJETO OK com {len(self.warnings)} avisos.")
        else:
            logger.error(f"❌ PROJETO COM PROBLEMAS: {len(self.errors)} erros, {len(self.warnings)} avisos.")
        
        if self.errors:
            logger.error("\n🔥 ERROS ENCONTRADOS:")
            for i, error in enumerate(self.errors, 1):
                logger.error(f"  {i}. {error}")
        
        if self.warnings:
            logger.warning("\n⚠️  AVISOS:")
            for i, warning in enumerate(self.warnings, 1):
                logger.warning(f"  {i}. {warning}")
        
        # Próximos passos
        if self.errors:
            logger.info("\n🔧 PRÓXIMOS PASSOS:")
            logger.info("1. Corrija os erros listados acima")
            logger.info("2. Execute novamente: python scripts/validate_project.py")
            logger.info("3. Se tudo estiver OK, execute os notebooks")
        else:
            logger.info("\n🚀 PROJETO PRONTO!")
            logger.info("1. Configure credenciais: cp .env.example .env")
            logger.info("2. Execute: jupyter lab")
            logger.info("3. Abra: notebooks/01_coleta.ipynb")

def main():
    """Função principal"""
    validator = ProjectValidator()
    validator.run_validation()
    
    # Exit code baseado nos resultados
    if validator.errors:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
