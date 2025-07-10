import snscrape.modules.twitter as sntwitter
import pandas as pd
import hashlib
from datetime import datetime
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class TwitterScraper:
    """Scraper para X/Twitter usando snscrape"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.twitter_config = config.get('twitter', {})
        
    def _hash_user_id(self, user_id: str) -> str:
        """Anonimizar user ID com hash SHA256"""
        return hashlib.sha256(str(user_id).encode()).hexdigest()[:12]
    
    def _build_query(self, keywords: List[str]) -> str:
        """Construir query de busca para snscrape"""
        # Combinar keywords com OR
        keyword_query = ' OR '.join([f'"{kw}"' for kw in keywords])
        
        # Adicionar filtros de idioma e país
        filters = []
        if self.twitter_config.get('lang'):
            filters.append(f"lang:{self.twitter_config['lang']}")
        if self.twitter_config.get('place_country'):
            filters.append(f"place_country:{self.twitter_config['place_country']}")
        
        # Construir query final
        query = f"({keyword_query})"
        if filters:
            query += " " + " ".join(filters)
            
        return query
    
    def scrape(self, keywords: List[str], limit: int = 800) -> List[Dict[str, Any]]:
        """
        Scrape tweets baseado nas keywords
        
        Args:
            keywords: Lista de palavras-chave para busca
            limit: Número máximo de tweets para coletar
            
        Returns:
            Lista de dicionários com dados dos tweets
        """
        query = self._build_query(keywords)
        logger.info(f"Scraping Twitter com query: {query}")
        logger.info(f"Limite: {limit} tweets")
        
        tweets_data = []
        
        try:
            # Usar snscrape para buscar tweets
            for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
                if i >= limit:
                    break
                
                # Filtrar retweets se configurado
                if self.twitter_config.get('exclude_retweets', True) and tweet.content.startswith('RT @'):
                    continue
                
                # Extrair dados do tweet
                tweet_data = {
                    'platform': 'twitter',
                    'id': tweet.id,
                    'text': tweet.content,
                    'timestamp': tweet.date,
                    'user_hash': self._hash_user_id(tweet.user.id),
                    'username': tweet.user.username,
                    'likes': tweet.likeCount or 0,
                    'retweets': tweet.retweetCount or 0,
                    'replies': tweet.replyCount or 0,
                    'url': tweet.url,
                    'lang': tweet.lang,
                    'collected_at': datetime.now()
                }
                
                tweets_data.append(tweet_data)
                
                # Log progresso a cada 100 tweets
                if (i + 1) % 100 == 0:
                    logger.info(f"Coletados {i + 1} tweets")
                    
        except Exception as e:
            logger.error(f"Erro ao scraping Twitter: {str(e)}")
            raise
        
        logger.info(f"Scraping concluído. Total: {len(tweets_data)} tweets")
        return tweets_data
    
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
            'total_tweets': len(df),
            'unique_users': df['user_hash'].nunique(),
            'avg_likes': df['likes'].mean(),
            'avg_retweets': df['retweets'].mean(),
            'date_range': {
                'start': df['timestamp'].min(),
                'end': df['timestamp'].max()
            },
            'languages': df['lang'].value_counts().to_dict()
        }