#!/usr/bin/env python3
"""
Script para limpeza e reset do projeto
Remove dados coletados e permite reiniciar análises
"""

import sys
import os
import shutil
from pathlib import Path
import argparse
import logging
from datetime import datetime

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class ProjectCleaner:
    """Limpador de dados do projeto"""
    
    def __init__(self, project_root=None):
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent
        self.backup_dir = self.project_root / "backups"
        
    def create_backup(self, data_dir):
        """Criar backup antes de limpar"""
        if not data_dir.exists() or not any(data_dir.iterdir()):
            logger.info(f"📂 Nenhum dado para backup em {data_dir}")
            return None
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"backup_{data_dir.name}_{timestamp}"
        backup_path = self.backup_dir / backup_name
        
        self.backup_dir.mkdir(exist_ok=True)
        shutil.copytree(data_dir, backup_path)
        logger.info(f"💾 Backup criado: {backup_path}")
        return backup_path
    
    def clean_raw_data(self, create_backup=True):
        """Limpar dados brutos coletados"""
        logger.info("🧹 Limpando dados brutos...")
        
        raw_dir = self.project_root / "data" / "raw"
        
        if create_backup:
            self.create_backup(raw_dir)
        
        if raw_dir.exists():
            # Listar o que será removido
            files_to_remove = []
            for item in raw_dir.rglob("*"):
                if item.is_file():
                    files_to_remove.append(item)
            
            if files_to_remove:
                logger.info(f"📄 Removendo {len(files_to_remove)} arquivos...")
                shutil.rmtree(raw_dir)
                raw_dir.mkdir(parents=True, exist_ok=True)
                logger.info("✅ Dados brutos removidos")
            else:
                logger.info("📂 Nenhum dado bruto para remover")
        else:
            logger.info("📂 Diretório de dados brutos não existe")
    
    def clean_processed_data(self, create_backup=True):
        """Limpar dados processados"""
        logger.info("🧹 Limpando dados processados...")
        
        processed_dir = self.project_root / "data" / "processed"
        
        if create_backup:
            self.create_backup(processed_dir)
        
        if processed_dir.exists():
            files_to_remove = []
            for item in processed_dir.rglob("*"):
                if item.is_file():
                    files_to_remove.append(item)
            
            if files_to_remove:
                logger.info(f"📄 Removendo {len(files_to_remove)} arquivos processados...")
                shutil.rmtree(processed_dir)
                processed_dir.mkdir(parents=True, exist_ok=True)
                logger.info("✅ Dados processados removidos")
            else:
                logger.info("📂 Nenhum dado processado para remover")
        else:
            logger.info("📂 Diretório de dados processados não existe")
    
    def clean_labeled_data(self, create_backup=True):
        """Limpar dados rotulados"""
        logger.info("🧹 Limpando dados rotulados...")
        
        labeled_dir = self.project_root / "data" / "labeled"
        
        if create_backup:
            self.create_backup(labeled_dir)
        
        if labeled_dir.exists():
            files_to_remove = []
            for item in labeled_dir.rglob("*"):
                if item.is_file():
                    files_to_remove.append(item)
            
            if files_to_remove:
                logger.info(f"📄 Removendo {len(files_to_remove)} arquivos rotulados...")
                shutil.rmtree(labeled_dir)
                labeled_dir.mkdir(parents=True, exist_ok=True)
                logger.info("✅ Dados rotulados removidos")
            else:
                logger.info("📂 Nenhum dado rotulado para remover")
        else:
            logger.info("📂 Diretório de dados rotulados não existe")
    
    def clean_models(self, create_backup=True):
        """Limpar modelos treinados"""
        logger.info("🧹 Limpando modelos...")
        
        models_dir = self.project_root / "models"
        
        if create_backup:
            self.create_backup(models_dir)
        
        if models_dir.exists():
            model_files = []
            for ext in ["*.pth", "*.pkl", "*.joblib", "*.model"]:
                model_files.extend(models_dir.rglob(ext))
            
            if model_files:
                logger.info(f"🤖 Removendo {len(model_files)} arquivos de modelo...")
                for model_file in model_files:
                    model_file.unlink()
                logger.info("✅ Modelos removidos")
            else:
                logger.info("🤖 Nenhum modelo para remover")
        else:
            logger.info("📂 Diretório de modelos não existe")
    
    def clean_logs(self):
        """Limpar logs"""
        logger.info("🧹 Limpando logs...")
        
        logs_dir = self.project_root / "logs"
        
        if logs_dir.exists():
            log_files = list(logs_dir.glob("*.log"))
            if log_files:
                logger.info(f"📋 Removendo {len(log_files)} arquivos de log...")
                for log_file in log_files:
                    log_file.unlink()
                logger.info("✅ Logs removidos")
            else:
                logger.info("📋 Nenhum log para remover")
        else:
            logger.info("📂 Diretório de logs não existe")
    
    def clean_cache(self):
        """Limpar cache Python"""
        logger.info("🧹 Limpando cache Python...")
        
        cache_dirs = []
        for item in self.project_root.rglob("__pycache__"):
            if item.is_dir():
                cache_dirs.append(item)
        
        for item in self.project_root.rglob("*.pyc"):
            if item.is_file():
                item.unlink()
        
        for cache_dir in cache_dirs:
            shutil.rmtree(cache_dir)
        
        if cache_dirs:
            logger.info(f"🗂️  Removidos {len(cache_dirs)} diretórios de cache")
        else:
            logger.info("🗂️  Nenhum cache para remover")
    
    def clean_results(self, create_backup=True):
        """Limpar resultados e figuras"""
        logger.info("🧹 Limpando resultados...")
        
        results_dir = self.project_root / "results"
        figures_dir = self.project_root / "figures"
        
        for dir_path in [results_dir, figures_dir]:
            if create_backup:
                self.create_backup(dir_path)
            
            if dir_path.exists():
                files = []
                for item in dir_path.rglob("*"):
                    if item.is_file():
                        files.append(item)
                
                if files:
                    logger.info(f"📊 Removendo {len(files)} arquivos de {dir_path.name}...")
                    shutil.rmtree(dir_path)
                    dir_path.mkdir(parents=True, exist_ok=True)
                    logger.info(f"✅ {dir_path.name} limpo")
                else:
                    logger.info(f"📂 Nenhum arquivo em {dir_path.name}")
    
    def reset_notebooks(self):
        """Limpar outputs dos notebooks"""
        logger.info("🧹 Limpando outputs dos notebooks...")
        
        notebooks_dir = self.project_root / "notebooks"
        
        if notebooks_dir.exists():
            try:
                import nbformat
                
                notebook_files = list(notebooks_dir.glob("*.ipynb"))
                for nb_file in notebook_files:
                    # Ler notebook
                    with open(nb_file, 'r', encoding='utf-8') as f:
                        nb = nbformat.read(f, as_version=4)
                    
                    # Limpar outputs
                    for cell in nb.cells:
                        if cell.cell_type == 'code':
                            cell.outputs = []
                            cell.execution_count = None
                    
                    # Salvar notebook limpo
                    with open(nb_file, 'w', encoding='utf-8') as f:
                        nbformat.write(nb, f)
                
                logger.info(f"📓 Outputs limpos de {len(notebook_files)} notebooks")
                
            except ImportError:
                logger.warning("📓 nbformat não disponível - notebooks não foram limpos")
        else:
            logger.info("📂 Diretório de notebooks não existe")
    
    def list_backups(self):
        """Listar backups disponíveis"""
        if not self.backup_dir.exists():
            logger.info("📂 Nenhum backup encontrado")
            return []
        
        backups = []
        for item in self.backup_dir.iterdir():
            if item.is_dir():
                stats = item.stat()
                size_mb = sum(f.stat().st_size for f in item.rglob('*') if f.is_file()) / (1024*1024)
                created = datetime.fromtimestamp(stats.st_ctime)
                backups.append({
                    'name': item.name,
                    'path': item,
                    'size_mb': size_mb,
                    'created': created
                })
        
        if backups:
            logger.info("💾 BACKUPS DISPONÍVEIS:")
            for backup in sorted(backups, key=lambda x: x['created'], reverse=True):
                logger.info(f"  - {backup['name']}: {backup['size_mb']:.1f} MB ({backup['created'].strftime('%Y-%m-%d %H:%M')})")
        else:
            logger.info("📂 Nenhum backup encontrado")
        
        return backups

def main():
    """Função principal"""
    parser = argparse.ArgumentParser(description='Limpar dados do projeto')
    parser.add_argument('--all', action='store_true', help='Limpar tudo')
    parser.add_argument('--raw', action='store_true', help='Limpar dados brutos')
    parser.add_argument('--processed', action='store_true', help='Limpar dados processados')
    parser.add_argument('--labeled', action='store_true', help='Limpar dados rotulados')
    parser.add_argument('--models', action='store_true', help='Limpar modelos')
    parser.add_argument('--logs', action='store_true', help='Limpar logs')
    parser.add_argument('--cache', action='store_true', help='Limpar cache Python')
    parser.add_argument('--results', action='store_true', help='Limpar resultados')
    parser.add_argument('--notebooks', action='store_true', help='Limpar outputs dos notebooks')
    parser.add_argument('--no-backup', action='store_true', help='Não criar backup')
    parser.add_argument('--list-backups', action='store_true', help='Listar backups')
    parser.add_argument('--project-root', type=str, help='Caminho para raiz do projeto')
    
    args = parser.parse_args()
    
    cleaner = ProjectCleaner(args.project_root)
    
    if args.list_backups:
        cleaner.list_backups()
        return
    
    create_backup = not args.no_backup
    
    if args.all:
        logger.info("🧹 LIMPEZA COMPLETA DO PROJETO")
        logger.info("="*50)
        cleaner.clean_raw_data(create_backup)
        cleaner.clean_processed_data(create_backup)
        cleaner.clean_labeled_data(create_backup)
        cleaner.clean_models(create_backup)
        cleaner.clean_logs()
        cleaner.clean_cache()
        cleaner.clean_results(create_backup)
        cleaner.reset_notebooks()
        logger.info("✅ LIMPEZA COMPLETA CONCLUÍDA")
        return
    
    # Limpezas específicas
    if args.raw:
        cleaner.clean_raw_data(create_backup)
    
    if args.processed:
        cleaner.clean_processed_data(create_backup)
    
    if args.labeled:
        cleaner.clean_labeled_data(create_backup)
    
    if args.models:
        cleaner.clean_models(create_backup)
    
    if args.logs:
        cleaner.clean_logs()
    
    if args.cache:
        cleaner.clean_cache()
    
    if args.results:
        cleaner.clean_results(create_backup)
    
    if args.notebooks:
        cleaner.reset_notebooks()
    
    # Se nenhuma opção específica foi escolhida, mostrar ajuda
    if not any([args.raw, args.processed, args.labeled, args.models, 
                args.logs, args.cache, args.results, args.notebooks]):
        parser.print_help()
        logger.info("\n💡 EXEMPLOS DE USO:")
        logger.info("  python scripts/clean_project.py --all                 # Limpar tudo")
        logger.info("  python scripts/clean_project.py --raw --processed    # Limpar dados")
        logger.info("  python scripts/clean_project.py --cache --logs       # Limpar temporários")
        logger.info("  python scripts/clean_project.py --list-backups       # Listar backups")

if __name__ == "__main__":
    main()
