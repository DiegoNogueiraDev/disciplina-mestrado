#!/usr/bin/env python3
"""
Script para treinar modelo baseline TF-IDF + Regressão Logística
"""

import argparse
import sys
import os
import logging
import pandas as pd
from pathlib import Path
import numpy as np
import json
from datetime import datetime

# Adicionar src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.config import load_config, setup_logging
from models.baseline import BaselineClassifier, create_sample_labeled_data

def save_metrics_to_json(metrics, output_path, model_info=None):
    """Salva métricas de validação em arquivo JSON"""
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'model_type': 'baseline_tfidf_logistic',
        'metrics': metrics,
        'model_info': model_info or {}
    }
    
    # Criar diretório se não existir
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    logging.getLogger(__name__).info(f"Métricas salvas em: {output_path}")

def main():
    """Função principal"""
    parser = argparse.ArgumentParser(description='Treinar modelo baseline')
    parser.add_argument('--config', type=str, default='config/topic.yaml',
                        help='Caminho para arquivo de configuração')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Caminho para dados processados (CSV)')
    parser.add_argument('--sample-size', type=int, default=600,
                        help='Tamanho da amostra para rotulagem simulada')
    parser.add_argument('--output-dir', type=str, default='models',
                        help='Diretório para salvar modelo')
    parser.add_argument('--cv-folds', type=int, default=5,
                        help='Número de folds para validação cruzada')
    parser.add_argument('--max-features', type=int, default=10000,
                        help='Número máximo de features TF-IDF')
    parser.add_argument('--save-metrics', action='store_true',
                        help='Salvar métricas em results/metrics.json')
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
        
        # Carregar dados
        logger.info(f"Carregando dados de: {args.data_path}")
        df = pd.read_csv(args.data_path)
        logger.info(f"Dados carregados: {len(df)} registros")
        
        # Verificar se existem rótulos
        if 'sentiment' not in df.columns:
            logger.info("Coluna 'sentiment' não encontrada. Criando rótulos simulados...")
            df = create_sample_labeled_data(df, args.sample_size)
            
            # Salvar dados rotulados
            labeled_path = Path(args.data_path).parent / f"labeled_sample_{args.sample_size}.csv"
            df.to_csv(labeled_path, index=False)
            logger.info(f"Dados rotulados salvos em: {labeled_path}")
        
        # Preparar dados
        baseline = BaselineClassifier(max_features=args.max_features)
        texts, labels = baseline.prepare_data(df)
        
        if len(texts) < 50:
            logger.error("Dados insuficientes para treinamento (mínimo: 50 amostras)")
            sys.exit(1)
        
        # Validação cruzada
        logger.info("=== VALIDAÇÃO CRUZADA ===")
        cv_results = baseline.cross_validate(texts, labels, cv=args.cv_folds)
        
        # Exibir resultados
        print(f"\n{'='*50}")
        print("RESULTADOS DA VALIDAÇÃO CRUZADA")
        print(f"{'='*50}")
        print(f"F1-macro: {cv_results['mean_f1']:.3f} (±{cv_results['std_f1']:.3f})")
        
        for metric, scores in cv_results['scores_detail'].items():
            print(f"{metric}: {scores.mean():.3f} (±{scores.std():.3f})")
        
        # Salvar métricas em JSON
        if args.save_metrics:
            metrics_path = Path(args.output_dir) / "metrics.json"
            save_metrics_to_json(cv_results, str(metrics_path), model_info={
                'sample_size': args.sample_size,
                'max_features': args.max_features,
                'cv_folds': args.cv_folds
            })
        
        # Treinar modelo final
        logger.info("=== TREINAMENTO FINAL ===")
        baseline.fit(texts, labels)
        
        # Feature importance
        logger.info("Analisando importância das features...")
        feature_importance = baseline.get_feature_importance(top_n=10)
        
        print(f"\n{'='*50}")
        print("TOP FEATURES POR CLASSE")
        print(f"{'='*50}")
        
        for class_name, features in feature_importance.items():
            print(f"\n{class_name.upper()}:")
            print("  Mais indicativas:")
            for feature, coef in features['positive'][:5]:
                print(f"    {feature}: {coef:.3f}")
            print("  Menos indicativas:")
            for feature, coef in features['negative'][:5]:
                print(f"    {feature}: {coef:.3f}")
        
        # Salvar modelo
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = output_dir / "baseline_tfidf_lr.pkl"
        baseline.save_model(str(model_path))
        
        # Salvar relatório
        report_path = output_dir / "baseline_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("RELATÓRIO DO MODELO BASELINE\n")
            f.write("="*50 + "\n\n")
            f.write(f"Dados de treinamento: {len(texts)} amostras\n")
            f.write(f"Features TF-IDF: {args.max_features}\n")
            f.write(f"N-gramas: (1, 2)\n\n")
            f.write("VALIDAÇÃO CRUZADA:\n")
            f.write(f"F1-macro: {cv_results['mean_f1']:.3f} (±{cv_results['std_f1']:.3f})\n\n")
            
            f.write("TOP FEATURES POR CLASSE:\n")
            for class_name, features in feature_importance.items():
                f.write(f"\n{class_name.upper()}:\n")
                f.write("  Mais indicativas:\n")
                for feature, coef in features['positive'][:10]:
                    f.write(f"    {feature}: {coef:.3f}\n")
        
        logger.info(f"Relatório salvo em: {report_path}")
        logger.info("Treinamento do baseline concluído com sucesso!")
        
    except Exception as e:
        logger.error(f"Erro durante o treinamento: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()