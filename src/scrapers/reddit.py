import praw
import pandas as pd
import hashlib
from datetime import datetime
from typing import List, Dict, Any
import logging
from langdetect import detect
import os

logger = logging.getLogger(__name__)

class RedditScraper:
    """Scraper para Reddit usando PRAW"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.reddit_config = config.get('reddit', {})
        
        # Inicializar cliente Reddit (read-only para scraping público)
        self.reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_ID', self.reddit_config.get('client_id', '')),
            client_secret=os.getenv('REDDIT_SECRET', self.reddit_config.get('client_secret', '')),
            user_agent=os.getenv('REDDIT_AGENT', self.reddit_config.get('user_agent', 'sentiment-analysis-bot/1.0')),
            check_for_async=False
        )
        
    def _hash_user_id(self, user_id: str) -> str:
        """Anonimizar user ID com hash SHA256"""
        return hashlib.sha256(str(user_id).encode()).hexdigest()[:12]
    
    def _is_portuguese(self, text: str) -> bool:
        """Detectar se texto está em português"""
        try:
            return detect(text) == 'pt'
        except:
            return False
    
    def _search_subreddit(self, subreddit_name: str, keywords: List[str], limit: int) -> List[Dict[str, Any]]:
        """Buscar posts em um subreddit específico"""
        subreddit = self.reddit.subreddit(subreddit_name)
        posts_data = []
        
        # Buscar por cada keyword
        for keyword in keywords:
            try:
                # Buscar posts recentes
                for submission in subreddit.search(keyword, sort='new', time_filter='month', limit=limit):
                    # Combinar título e texto
                    full_text = submission.title
                    if submission.selftext:
                        full_text += " " + submission.selftext
                    
                    # Filtrar por idioma
                    if not self._is_portuguese(full_text):
                        continue
                    
                    # Filtrar por comprimento
                    if len(full_text) < self.config.get('filters', {}).get('min_length', 10):
                        continue
                    
                    post_data = {
                        'platform': 'reddit',
                        'id': submission.id,
                        'text': full_text,
                        'title': submission.title,
                        'selftext': submission.selftext,
                        'timestamp': datetime.fromtimestamp(submission.created_utc),
                        'user_hash': self._hash_user_id(submission.author.name if submission.author else 'deleted'),
                        'username': submission.author.name if submission.author else 'deleted',
                        'score': submission.score,
                        'upvote_ratio': submission.upvote_ratio,
                        'num_comments': submission.num_comments,
                        'subreddit': subreddit_name,
                        'url': f"https://reddit.com{submission.permalink}",
                        'collected_at': datetime.now()
                    }
                    
                    posts_data.append(post_data)
                    
                    if len(posts_data) >= limit:
                        break
                        
            except Exception as e:
                logger.error(f"Erro ao buscar em r/{subreddit_name} com keyword '{keyword}': {str(e)}")
                continue
        
        return posts_data
    
    def scrape(self, keywords: List[str], limit: int = 200) -> List[Dict[str, Any]]:
        """
        Scrape posts do Reddit baseado nas keywords
        
        Args:
            keywords: Lista de palavras-chave para busca
            limit: Número máximo de posts para coletar
            
        Returns:
            Lista de dicionários com dados dos posts
        """
        subreddits = self.reddit_config.get('subreddits', ['brasil', 'politica', 'brasilivre'])
        logger.info(f"Scraping Reddit nos subreddits: {subreddits}")
        logger.info(f"Keywords: {keywords}")
        logger.info(f"Limite total: {limit} posts")
        
        all_posts = []
        posts_per_subreddit = limit // len(subreddits)
        
        for subreddit_name in subreddits:
            logger.info(f"Buscando em r/{subreddit_name}...")
            
            try:
                posts = self._search_subreddit(subreddit_name, keywords, posts_per_subreddit)
                all_posts.extend(posts)
                logger.info(f"Coletados {len(posts)} posts de r/{subreddit_name}")
                
            except Exception as e:
                logger.error(f"Erro ao scraping r/{subreddit_name}: {str(e)}")
                continue
        
        # Remover duplicatas baseado no ID
        unique_posts = []
        seen_ids = set()
        
        for post in all_posts:
            if post['id'] not in seen_ids:
                unique_posts.append(post)
                seen_ids.add(post['id'])
        
        # Ordenar por score (mais relevantes primeiro)
        unique_posts.sort(key=lambda x: x['score'], reverse=True)
        
        # Limitar ao número desejado
        final_posts = unique_posts[:limit]
        
        logger.info(f"Scraping concluído. Total: {len(final_posts)} posts únicos")
        return final_posts
    
    def save_to_csv(self, data: List[Dict[str, Any]], filepath: str) -> None:
        """Salvar dados em CSV"""
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        logger.info(f"Dados salvos em: {filepath}")
    
    def get_stats(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Obter estatísticas dos dados coletados"""
        if not data:
            return {}
        
        df = pd.DataFrame(data)
        return {
            'total_posts': len(df),
            'unique_users': df['user_hash'].nunique(),
            'avg_score': df['score'].mean(),
            'avg_comments': df['num_comments'].mean(),
            'subreddits': df['subreddit'].value_counts().to_dict(),
            'date_range': {
                'start': df['timestamp'].min(),
                'end': df['timestamp'].max()
            },
            'score_distribution': {
                'min': df['score'].min(),
                'max': df['score'].max(),
                'median': df['score'].median()
            }
        }