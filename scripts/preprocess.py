#!/usr/bin/env python3
"""
Script para pré-processamento dos dados coletados
"""

import argparse
import sys
import os
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
import hashlib

# Adicionar src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.config import load_config, setup_logging
from preprocessing.cleaner import TextCleaner

def hash_user_id(user_id):
    """Cria hash do user_id para conformidade LGPD"""
    if pd.isna(user_id):
        return None
    return hashlib.sha256(str(user_id).encode()).hexdigest()

def save_to_parquet(df, output_path):
    """Salva DataFrame em formato Parquet usando DuckDB se disponível"""
    try:
        import duckdb
        
        # Usar DuckDB para salvar parquet de forma eficiente
        conn = duckdb.connect()
        conn.execute(f"COPY (SELECT * FROM df) TO '{output_path}' (FORMAT PARQUET)")
        conn.close()
        
        logging.getLogger(__name__).info(f"Dados salvos em Parquet via DuckDB: {output_path}")
        
    except ImportError:
        # Fallback para pandas
        df.to_parquet(output_path, index=False, compression='snappy')
        logging.getLogger(__name__).info(f"Dados salvos em Parquet via pandas: {output_path}")

def main():
    """Função principal"""
    parser = argparse.ArgumentParser(description='Pré-processar dados coletados')
    parser.add_argument('--config', type=str, default='config/topic.yaml',
                        help='Caminho para arquivo de configuração')
    parser.add_argument('--input-dir', type=str, default='data/raw',
                        help='Diretório com dados brutos')
    parser.add_argument('--output-dir', type=str, default='data/processed',
                        help='Diretório para dados processados')
    parser.add_argument('--pattern', type=str, default='*.csv',
                        help='Padrão para buscar arquivos')
    parser.add_argument('--format', type=str, choices=['csv', 'parquet'], 
                        default='parquet', help='Formato de saída')
    parser.add_argument('--hash-users', action='store_true',
                        help='Hash user IDs para conformidade LGPD')
    parser.add_argument('--combine', action='store_true',
                        help='Combinar todos os arquivos em um único dataset')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Modo verboso')
    
    args = parser.parse_args()
    
    # Configurar logging
    log_level = 'DEBUG' if args.verbose else 'INFO'
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Carregar configuração
        config = load_config(args.config)
        logger.info(f"Configuração carregada: {args.config}")
        
        # Inicializar limpador de texto
        cleaner = TextCleaner(config)
        
        # Criar diretório de saída
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Buscar arquivos para processar
        input_dir = Path(args.input_dir)
        csv_files = list(input_dir.glob(args.pattern))
        
        if not csv_files:
            logger.error(f"Nenhum arquivo encontrado em {input_dir} com padrão {args.pattern}")
            sys.exit(1)
        
        logger.info(f"Encontrados {len(csv_files)} arquivos para processar")
        
        processed_dfs = []
        
        for csv_file in csv_files:
            logger.info(f"Processando: {csv_file}")
            
            try:
                # Carregar dados
                df = pd.read_csv(csv_file)
                logger.info(f"Carregados {len(df)} registros")
                
                # Determinar coluna de texto
                text_column = 'text'
                if 'text' not in df.columns:
                    logger.error(f"Coluna 'text' não encontrada em {csv_file}")
                    continue
                
                # Processar textos
                processed_df = cleaner.process_dataframe(df, text_column)
                
                # Estatísticas
                stats = cleaner.get_processing_stats(processed_df)
                logger.info(f"Estatísticas - Total: {stats['total_texts']}, Válidos: {stats['valid_texts']}")
                
                # Hash de user IDs se solicitado
                if args.hash_users and 'user_id' in df.columns:
                    logger.info("Aplicando hash em user IDs")
                    processed_df['user_id'] = processed_df['user_id'].apply(hash_user_id)
                
                if args.combine:
                    # Adicionar à lista para combinar
                    processed_dfs.append(processed_df)
                else:
                    # Salvar arquivo individual
                    output_file = output_dir / f"processed_{csv_file.stem}.parquet"
                    save_to_parquet(processed_df, output_file)
                    logger.info(f"Dados processados salvos em: {output_file}")
                
            except Exception as e:
                logger.error(f"Erro ao processar {csv_file}: {str(e)}")
                continue
        
        # Combinar todos os dados se solicitado
        if args.combine and processed_dfs:
            logger.info("Combinando todos os datasets...")
            combined_df = pd.concat(processed_dfs, ignore_index=True)
            
            # Remover duplicatas baseado no texto
            initial_count = len(combined_df)
            combined_df = combined_df.drop_duplicates(subset=['lemmatized'])
            final_count = len(combined_df)
            
            logger.info(f"Removidas {initial_count - final_count} duplicatas")
            
            # Salvar dataset combinado
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = output_dir / f"combined_processed_{timestamp}.parquet"
            save_to_parquet(combined_df, output_file)
            
            logger.info(f"Dataset combinado salvo em: {output_file}")
            logger.info(f"Total de registros: {len(combined_df)}")
            
            # Estatísticas finais
            platform_counts = combined_df['platform'].value_counts()
            logger.info(f"Distribuição por plataforma: {platform_counts.to_dict()}")
        
        logger.info("Pré-processamento concluído com sucesso!")
        
    except Exception as e:
        logger.error(f"Erro durante o pré-processamento: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()