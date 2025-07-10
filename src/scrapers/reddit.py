import praw
import pandas as pd
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging
from langdetect import detect
import os
import time
import random

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
        """Buscar posts em um subreddit específico com múltiplas estratégias"""
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            posts_data = []
            seen_ids = set()
            
            # Estratégias de busca
            search_methods = self.reddit_config.get('search_methods', ['hot', 'top', 'new'])
            time_filter = self.reddit_config.get('time_filter', 'month')
            
            # Distribuir limite entre métodos de busca
            limit_per_method = max(1, limit // len(search_methods))
            
            for method in search_methods:
                logger.info(f"  Buscando posts '{method}' em r/{subreddit_name}")
                
                try:
                    # Busca por keywords
                    for keyword in keywords:
                        try:
                            if method == 'hot':
                                submissions = subreddit.search(keyword, sort='hot', time_filter=time_filter, limit=limit_per_method)
                            elif method == 'top':
                                submissions = subreddit.search(keyword, sort='top', time_filter=time_filter, limit=limit_per_method)
                            elif method == 'new':
                                submissions = subreddit.search(keyword, sort='new', time_filter=time_filter, limit=limit_per_method)
                            else:
                                submissions = subreddit.search(keyword, sort='relevance', time_filter=time_filter, limit=limit_per_method)
                            
                            for submission in submissions:
                                # Evitar duplicatas
                                if submission.id in seen_ids:
                                    continue
                                seen_ids.add(submission.id)
                                
                                # Combinar título e texto
                                full_text = submission.title
                                if submission.selftext:
                                    full_text += " " + submission.selftext
                                
                                # Filtros básicos
                                if len(full_text) < self.config.get('filters', {}).get('min_length', 15):
                                    continue
                                
                                if len(full_text) > self.config.get('filters', {}).get('max_length', 500):
                                    continue
                                
                                # Filtrar por score mínimo
                                min_score = self.config.get('filters', {}).get('min_score', 0)
                                if submission.score < min_score:
                                    continue
                                
                                # Verificar se é português (apenas para subreddits internacionais)
                                if subreddit_name in ['politics', 'worldnews', 'news', 'economics', 'geopolitics', 'conservative', 'liberal']:
                                    if not self._is_portuguese(full_text):
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
                                    'search_method': method,
                                    'search_keyword': keyword,
                                    'collected_at': datetime.now()
                                }
                                
                                posts_data.append(post_data)
                                
                                if len(posts_data) >= limit:
                                    break
                                    
                        except Exception as e:
                            logger.warning(f"Erro ao buscar '{keyword}' em r/{subreddit_name}: {str(e)}")
                            continue
                        
                        # Rate limiting
                        time.sleep(random.uniform(0.1, 0.3))
                        
                    if len(posts_data) >= limit:
                        break
                        
                except Exception as e:
                    logger.warning(f"Erro no método '{method}' em r/{subreddit_name}: {str(e)}")
                    continue
                    
            # Busca adicional nos posts populares do subreddit (sem keyword)
            if len(posts_data) < limit:
                logger.info(f"  Buscando posts populares gerais em r/{subreddit_name}")
                try:
                    remaining_limit = limit - len(posts_data)
                    for submission in subreddit.hot(limit=min(remaining_limit, 100)):
                        if submission.id in seen_ids:
                            continue
                        seen_ids.add(submission.id)
                        
                        full_text = submission.title
                        if submission.selftext:
                            full_text += " " + submission.selftext
                        
                        # Verificar se contém alguma keyword
                        contains_keyword = any(keyword.lower() in full_text.lower() for keyword in keywords)
                        if not contains_keyword:
                            continue
                            
                        # Aplicar filtros
                        if len(full_text) < self.config.get('filters', {}).get('min_length', 15):
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
                            'search_method': 'hot_general',
                            'search_keyword': 'general',
                            'collected_at': datetime.now()
                        }
                        
                        posts_data.append(post_data)
                        
                        if len(posts_data) >= limit:
                            break
                            
                except Exception as e:
                    logger.warning(f"Erro ao buscar posts populares em r/{subreddit_name}: {str(e)}")
            
            return posts_data
            
        except Exception as e:
            logger.error(f"Erro ao acessar r/{subreddit_name}: {str(e)}")
            return []
    
    def scrape(self, keywords: List[str], limit: int = 200) -> List[Dict[str, Any]]:
        """
        Scrape posts do Reddit baseado nas keywords com estratégia otimizada para alto volume
        
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
        
        # Estratégia adaptativa: dar mais foco aos subreddits principais
        priority_subreddits = ['brasil', 'brasilivre', 'politica', 'worldnews', 'politics']
        
        # Distribuir limite de forma inteligente
        total_subreddits = len(subreddits)
        
        for i, subreddit_name in enumerate(subreddits):
            # Dar mais quota aos subreddits prioritários
            if subreddit_name in priority_subreddits:
                subreddit_limit = int(limit * 0.15)  # 15% para cada subreddit prioritário
            else:
                remaining_quota = limit - (len(priority_subreddits) * int(limit * 0.15))
                non_priority_count = total_subreddits - len([s for s in subreddits if s in priority_subreddits])
                subreddit_limit = max(10, remaining_quota // max(1, non_priority_count))
            
            logger.info(f"Buscando em r/{subreddit_name} (quota: {subreddit_limit})...")
            
            try:
                posts = self._search_subreddit(subreddit_name, keywords, subreddit_limit)
                all_posts.extend(posts)
                logger.info(f"Coletados {len(posts)} posts de r/{subreddit_name}")
                
                # Rate limiting entre subreddits
                time.sleep(random.uniform(0.5, 1.0))
                
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
        
        # Ordenar por relevância: combinar score, comentários e data
        def calculate_relevance(post):
            score_weight = 0.4
            comments_weight = 0.3
            recency_weight = 0.3
            
            # Normalizar score (0-1)
            max_score = max(p['score'] for p in unique_posts) if unique_posts else 1
            normalized_score = min(1.0, max(0.0, post['score'] / max_score))
            
            # Normalizar comentários (0-1)
            max_comments = max(p['num_comments'] for p in unique_posts) if unique_posts else 1
            normalized_comments = min(1.0, post['num_comments'] / max_comments)
            
            # Normalizar recência (posts mais recentes = maior peso)
            now = datetime.now()
            days_ago = (now - post['timestamp']).days
            normalized_recency = max(0.0, 1.0 - (days_ago / 30))  # 30 dias = 0
            
            return (score_weight * normalized_score + 
                   comments_weight * normalized_comments + 
                   recency_weight * normalized_recency)
        
        unique_posts.sort(key=calculate_relevance, reverse=True)
        
        # Limitar ao número desejado
        final_posts = unique_posts[:limit]
        
        logger.info(f"Scraping concluído. Total: {len(final_posts)} posts únicos")
        logger.info(f"Distribuição por subreddit: {self._get_subreddit_distribution(final_posts)}")
        
        return final_posts
    
    def _get_subreddit_distribution(self, posts: List[Dict[str, Any]]) -> Dict[str, int]:
        """Obter distribuição de posts por subreddit"""
        distribution = {}
        for post in posts:
            subreddit = post['subreddit']
            distribution[subreddit] = distribution.get(subreddit, 0) + 1
        return distribution
    
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