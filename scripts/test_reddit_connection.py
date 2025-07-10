#!/usr/bin/env python3
"""
Script para testar conectividade com a API do Reddit.
Diagnóstico rápido de credenciais PRAW.
"""

import os
import sys
from pathlib import Path

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import praw
from dotenv import load_dotenv

def test_reddit_connection():
    """Testa conexão com Reddit API usando credenciais do .env"""
    
    # Carregar variáveis de ambiente
    load_dotenv()
    
    required_vars = [
        'REDDIT_CLIENT_ID',
        'REDDIT_CLIENT_SECRET', 
        'REDDIT_USER_AGENT'
    ]
    
    # Verificar se todas as variáveis estão definidas
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ ERRO: Variáveis de ambiente não configuradas:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\n🔧 Configure o arquivo .env com suas credenciais do Reddit")
        return False
    
    try:
        # Inicializar cliente Reddit
        reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent=os.getenv('REDDIT_USER_AGENT')
        )
        
        # Teste básico: obter 1 post do subreddit popular
        print("🔄 Testando conexão com Reddit API...")
        
        subreddit = reddit.subreddit('popular')
        post = next(subreddit.hot(limit=1))
        
        print(f"✅ SUCESSO: Conexão estabelecida!")
        print(f"   📝 Post teste: {post.title[:50]}...")
        print(f"   🔗 URL: {post.url}")
        print(f"   👤 User-Agent: {os.getenv('REDDIT_USER_AGENT')}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERRO na conexão: {str(e)}")
        print("\n🔧 Possíveis soluções:")
        print("   1. Verificar credenciais no .env")
        print("   2. Verificar se o app Reddit está configurado corretamente")
        print("   3. Verificar conectividade com a internet")
        return False

if __name__ == "__main__":
    print("🧪 Teste de Conectividade - Reddit API")
    print("=" * 40)
    
    success = test_reddit_connection()
    
    if success:
        print("\n🎉 Reddit API funcionando corretamente!")
        sys.exit(0)
    else:
        print("\n💥 Falha na conexão com Reddit API")
        sys.exit(1)
