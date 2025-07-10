#!/usr/bin/env python3
"""
Script principal para coleta de dados do X/Twitter e Reddit
"""

import argparse
import sys
import os
import logging
from datetime import datetime
from pathlib import Path

# Adicionar src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.config import load_config, setup_logging
from scrapers.twitter import TwitterScraper
from scrapers.twitter_selenium import TwitterSeleniumScraper
from scrapers.reddit import RedditScraper

def main():
    """Função principal"""
    parser = argparse.ArgumentParser(description='Coletar dados do X/Twitter e Reddit')
    parser.add_argument('--config', type=str, default='config/topic.yaml',
                        help='Caminho para arquivo de configuração')
    parser.add_argument('--limit-twitter', type=int, default=None,
                        help='Limite de tweets para coletar')
    parser.add_argument('--limit-reddit', type=int, default=None,
                        help='Limite de posts do Reddit para coletar')
    parser.add_argument('--topic', type=str, default=None,
                        help='Tópico específico para buscar')
    parser.add_argument('--output-dir', type=str, default='data/raw',
                        help='Diretório para salvar dados coletados')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Modo verboso')
    parser.add_argument('--use-selenium', action='store_true',
                        help='Usar Selenium para Twitter em vez de snscrape')
    
    args = parser.parse_args()
    
    # Configurar logging
    log_level = 'DEBUG' if args.verbose else 'INFO'
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Carregar configuração
        config = load_config(args.config)
        logger.info(f"Configuração carregada: {args.config}")
        
        # Usar limites dos argumentos ou da configuração
        twitter_limit = args.limit_twitter or config.get('limits', {}).get('twitter', 800)
        reddit_limit = args.limit_reddit or config.get('limits', {}).get('reddit', 200)
        
        # Usar tópico do argumento ou da configuração
        topic = args.topic or config.get('topic', 'default')
        keywords = config.get('keywords', [topic])
        
        logger.info(f"Iniciando coleta para tópico: {topic}")
        logger.info(f"Keywords: {keywords}")
        logger.info(f"Limites - Twitter: {twitter_limit}, Reddit: {reddit_limit}")
        
        # Criar diretório de saída
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Timestamp para nomes de arquivo
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Coletar dados do Twitter
        if twitter_limit > 0:
            logger.info("Iniciando coleta do Twitter...")
            try:
                if args.use_selenium:
                    logger.info("Usando Selenium para coleta do Twitter")
                    twitter_scraper = TwitterSeleniumScraper(config)
                else:
                    logger.info("Usando snscrape para coleta do Twitter")
                    twitter_scraper = TwitterScraper(config)
                
                twitter_data = twitter_scraper.scrape(keywords, twitter_limit)
                
                if twitter_data:
                    twitter_file = output_dir / f"twitter_{topic}_{timestamp}.csv"
                    twitter_scraper.save_to_csv(twitter_data, twitter_file)
                    
                    # Estatísticas
                    stats = twitter_scraper.get_stats(twitter_data)
                    logger.info(f"Twitter - Coletados: {stats['total_tweets']} tweets")
                    logger.info(f"Twitter - Usuários únicos: {stats['unique_users']}")
                    logger.info(f"Twitter - Média de likes: {stats['avg_likes']:.1f}")
                else:
                    logger.warning("Nenhum dado coletado do Twitter")
            except Exception as e:
                logger.error(f"Erro na coleta do Twitter: {str(e)}")
                # Se snscrape falhar, tentar Selenium como fallback
                if not args.use_selenium:
                    logger.info("Tentando fallback com Selenium...")
                    try:
                        twitter_scraper = TwitterSeleniumScraper(config)
                        twitter_data = twitter_scraper.scrape(keywords, twitter_limit)
                        
                        if twitter_data:
                            twitter_file = output_dir / f"twitter_{topic}_{timestamp}.csv"
                            twitter_scraper.save_to_csv(twitter_data, twitter_file)
                            
                            # Estatísticas
                            stats = twitter_scraper.get_stats(twitter_data)
                            logger.info(f"Twitter (Selenium) - Coletados: {stats['total_tweets']} tweets")
                            logger.info(f"Twitter (Selenium) - Usuários únicos: {stats['unique_users']}")
                            logger.info(f"Twitter (Selenium) - Média de likes: {stats['avg_likes']:.1f}")
                        else:
                            logger.warning("Nenhum dado coletado do Twitter (Selenium)")
                    except Exception as selenium_error:
                        logger.error(f"Erro no fallback do Selenium: {str(selenium_error)}")
        else:
            logger.info("Coleta do Twitter desabilitada (limit=0)")
        
        # Coletar dados do Reddit
        if reddit_limit > 0:
            logger.info("Iniciando coleta do Reddit...")
            try:
                reddit_scraper = RedditScraper(config)
                reddit_data = reddit_scraper.scrape(keywords, reddit_limit)
                
                if reddit_data:
                    reddit_file = output_dir / f"reddit_{topic}_{timestamp}.csv"
                    reddit_scraper.save_to_csv(reddit_data, reddit_file)
                    
                    # Estatísticas
                    stats = reddit_scraper.get_stats(reddit_data)
                    logger.info(f"Reddit - Coletados: {stats['total_posts']} posts")
                    logger.info(f"Reddit - Usuários únicos: {stats['unique_users']}")
                    logger.info(f"Reddit - Score médio: {stats['avg_score']:.1f}")
                else:
                    logger.warning("Nenhum dado coletado do Reddit")
            except Exception as e:
                logger.error(f"Erro na coleta do Reddit: {str(e)}")
        else:
            logger.info("Coleta do Reddit desabilitada (limit=0)")
        
        logger.info("Coleta concluída com sucesso!")
        
    except Exception as e:
        logger.error(f"Erro durante a coleta: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()