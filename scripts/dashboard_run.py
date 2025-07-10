#!/usr/bin/env python3
"""
Script para executar o dashboard
"""

import argparse
import sys
import os
import logging

# Adicionar src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.config import setup_logging
from dashboard.app import create_dashboard_app

def main():
    """Função principal"""
    parser = argparse.ArgumentParser(description='Executar dashboard de análise de sentimento')
    parser.add_argument('--data-path', type=str, default='data/output',
                        help='Caminho para os dados de saída')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host para o servidor')
    parser.add_argument('--port', type=int, default=8050,
                        help='Porta para o servidor')
    parser.add_argument('--debug', action='store_true',
                        help='Executar em modo debug')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Modo verboso')
    
    args = parser.parse_args()
    
    # Configurar logging
    log_level = 'DEBUG' if args.verbose else 'INFO'
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Iniciando dashboard...")
        logger.info(f"Dados: {args.data_path}")
        logger.info(f"Servidor: {args.host}:{args.port}")
        
        # Criar aplicação
        app = create_dashboard_app(args.data_path)
        
        # Executar servidor
        app.run(
            debug=args.debug,
            host=args.host,
            port=args.port
        )
        
    except Exception as e:
        logger.error(f"Erro ao executar dashboard: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()