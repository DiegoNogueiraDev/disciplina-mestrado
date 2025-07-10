#!/usr/bin/env python3
"""
Script para inferência em lote usando modelo treinado
"""

import argparse
import sys
import os
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime

# Adicionar src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.config import load_config, setup_logging
from models.inference import SentimentInference

def main():
    """Função principal"""
    parser = argparse.ArgumentParser(description='Fazer inferência de sentimento em lote')
    parser.add_argument('--config', type=str, default='config/topic.yaml',
                        help='Caminho para arquivo de configuração')
    parser.add_argument('--input-dir', type=str, default='data/processed',
                        help='Diretório com dados processados')
    parser.add_argument('--output-dir', type=str, default='data/output',
                        help='Diretório para resultados')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Caminho para o modelo treinado (.pth)')
    parser.add_argument('--fasttext-path', type=str, required=True,
                        help='Caminho para o modelo FastText (.bin)')
    parser.add_argument('--pattern', type=str, default='*.csv',
                        help='Padrão para buscar arquivos')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Tamanho do lote para inferência')
    parser.add_argument('--text-column', type=str, default='lemmatized',
                        help='Coluna com texto para classificar')
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
        
        # Verificar se modelos existem
        if not os.path.exists(args.model_path):
            logger.error(f"Modelo não encontrado: {args.model_path}")
            sys.exit(1)
        
        if not os.path.exists(args.fasttext_path):
            logger.error(f"Modelo FastText não encontrado: {args.fasttext_path}")
            sys.exit(1)
        
        # Inicializar modelo de inferência
        logger.info("Carregando modelos...")
        inference = SentimentInference(args.model_path, args.fasttext_path)
        logger.info("Modelos carregados com sucesso!")
        
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
        
        all_results = []
        
        for csv_file in csv_files:
            logger.info(f"Processando: {csv_file}")
            
            try:
                # Carregar dados
                df = pd.read_csv(csv_file)
                logger.info(f"Carregados {len(df)} registros")
                
                # Verificar se coluna de texto existe
                if args.text_column not in df.columns:
                    logger.error(f"Coluna '{args.text_column}' não encontrada em {csv_file}")
                    continue
                
                # Filtrar registros com texto válido
                valid_df = df[df[args.text_column].notna() & (df[args.text_column] != '')].copy()
                
                if valid_df.empty:
                    logger.warning(f"Nenhum texto válido encontrado em {csv_file}")
                    continue
                
                logger.info(f"Registros válidos: {len(valid_df)}")
                
                # Fazer inferência
                result_df = inference.predict_dataframe(valid_df, args.text_column)
                
                # Estatísticas
                stats = inference.get_prediction_stats(result_df)
                logger.info(f"Predições - Total: {stats['total_predictions']}")
                logger.info(f"Distribuição: {stats['sentiment_distribution']}")
                logger.info(f"Confiança média: {stats['avg_confidence']:.3f}")
                
                # Salvar resultados
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_file = output_dir / f"sentiment_{csv_file.stem}_{timestamp}.csv"
                result_df.to_csv(output_file, index=False)
                
                logger.info(f"Resultados salvos em: {output_file}")
                
                # Adicionar à lista para estatísticas gerais
                all_results.append(result_df)
                
            except Exception as e:
                logger.error(f"Erro ao processar {csv_file}: {str(e)}")
                continue
        
        # Estatísticas gerais
        if all_results:
            combined_df = pd.concat(all_results, ignore_index=True)
            
            logger.info("=== ESTATÍSTICAS GERAIS ===")
            logger.info(f"Total de registros processados: {len(combined_df)}")
            
            sentiment_counts = combined_df['predicted_sentiment'].value_counts()
            total = len(combined_df)
            
            for sentiment, count in sentiment_counts.items():
                percentage = (count / total) * 100
                logger.info(f"{sentiment.capitalize()}: {count} ({percentage:.1f}%)")
            
            # Estatísticas por plataforma
            if 'platform' in combined_df.columns:
                logger.info("\n=== POR PLATAFORMA ===")
                platform_stats = combined_df.groupby(['platform', 'predicted_sentiment']).size().unstack(fill_value=0)
                
                for platform in platform_stats.index:
                    platform_total = platform_stats.loc[platform].sum()
                    logger.info(f"\n{platform.upper()}: {platform_total} posts")
                    
                    for sentiment in platform_stats.columns:
                        count = platform_stats.loc[platform, sentiment]
                        percentage = (count / platform_total) * 100 if platform_total > 0 else 0
                        logger.info(f"  {sentiment}: {count} ({percentage:.1f}%)")
            
            # Salvar dataset combinado final
            final_output = output_dir / f"final_sentiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            combined_df.to_csv(final_output, index=False)
            logger.info(f"\nDataset final salvo em: {final_output}")
        
        logger.info("Inferência concluída com sucesso!")
        
    except Exception as e:
        logger.error(f"Erro durante a inferência: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()