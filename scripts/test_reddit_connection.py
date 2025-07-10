#!/usr/bin/env python3
"""
Script para testar conectividade com Reddit API
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import praw
from dotenv import load_dotenv
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_reddit_connection():
    """Testar conexão com Reddit API"""
    
    # Carregar variáveis de ambiente
    load_dotenv()
    
    # Verificar se credenciais estão configuradas
    required_vars = ['REDDIT_ID', 'REDDIT_SECRET', 'REDDIT_AGENT']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"❌ Variáveis de ambiente faltando: {missing_vars}")
        logger.info("📋 Configure as credenciais no arquivo .env:")
        logger.info("   1. Copie .env.example para .env")
        logger.info("   2. Acesse https://www.reddit.com/prefs/apps")
        logger.info("   3. Crie um novo app (tipo 'script')")
        logger.info("   4. Adicione as credenciais ao arquivo .env")
        return False
    
    try:
        # Inicializar cliente Reddit
        reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_ID'),
            client_secret=os.getenv('REDDIT_SECRET'),
            user_agent=os.getenv('REDDIT_AGENT'),
            check_for_async=False
        )
        
        logger.info("🔗 Testando conexão com Reddit API...")
        
        # Testar acesso básico
        subreddit = reddit.subreddit('brasil')
        logger.info(f"✅ Conectado ao r/brasil: {subreddit.display_name}")
        
        # Testar busca
        logger.info("🔍 Testando busca...")
        results = list(subreddit.search('trump', limit=1))
        
        if results:
            post = results[0]
            logger.info(f"✅ Busca funcionando: '{post.title[:50]}...'")
        else:
            logger.info("⚠️ Nenhum resultado encontrado para 'trump'")
        
        # Testar rate limits
        logger.info("📊 Informações da API:")
        logger.info(f"   Rate limit: {reddit.auth.limits}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro na conexão: {e}")
        return False

def main():
    """Função principal"""
    logger.info("🚀 Testando conectividade com Reddit API")
    
    if test_reddit_connection():
        logger.info("✅ Conexão com Reddit API funcionando!")
        logger.info("🎯 Pronto para coletar dados sobre Trump/Brasil")
    else:
        logger.error("❌ Falha na conexão com Reddit API")
        logger.info("📚 Verifique as instruções no README.md")

if __name__ == '__main__':
    main()