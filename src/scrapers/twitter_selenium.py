import time
import pandas as pd
import hashlib
import os
from datetime import datetime
from typing import List, Dict, Any
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import re

logger = logging.getLogger(__name__)

class TwitterSeleniumScraper:
    """Scraper para X/Twitter usando Selenium"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.twitter_config = config.get('twitter', {})
        self.driver = None
        
    def _setup_driver(self):
        """Configurar driver do Selenium"""
        # Verificar se Chrome/Chromium está disponível
        import shutil
        
        chrome_options = Options()
        chrome_options.add_argument('--headless')  # Executar sem interface gráfica
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        # Tentar encontrar Chrome/Chromium
        possible_browsers = [
            '/usr/bin/google-chrome',
            '/usr/bin/chromium-browser',
            '/usr/bin/chromium',
            '/snap/bin/chromium',
            shutil.which('google-chrome'),
            shutil.which('chromium-browser'),
            shutil.which('chromium')
        ]
        
        browser_path = None
        for path in possible_browsers:
            if path and shutil.which(path) or (path and os.path.exists(path)):
                browser_path = path
                break
        
        if browser_path:
            chrome_options.binary_location = browser_path
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.implicitly_wait(10)
            logger.info("Driver do Chrome configurado com sucesso")
        except Exception as e:
            logger.error(f"Erro ao configurar driver: {e}")
            logger.error("Chrome/Chromium não encontrado. Por favor, instale o Chrome ou Chromium.")
            raise ValueError("Browser não encontrado. Instale Chrome ou Chromium para usar o scraper Selenium.")
    
    def _close_driver(self):
        """Fechar driver do Selenium"""
        if self.driver:
            self.driver.quit()
            self.driver = None
    
    def _hash_user_id(self, user_id: str) -> str:
        """Anonimizar user ID com hash SHA256"""
        return hashlib.sha256(str(user_id).encode()).hexdigest()[:12]
    
    def _build_search_url(self, keywords: List[str]) -> str:
        """Construir URL de busca para Twitter"""
        # Combinar keywords com OR
        query = ' OR '.join([f'"{kw}"' for kw in keywords])
        
        # Adicionar filtros de idioma
        if self.twitter_config.get('lang'):
            query += f" lang:{self.twitter_config['lang']}"
        
        # URL encode da query
        import urllib.parse
        encoded_query = urllib.parse.quote(query)
        
        return f"https://twitter.com/search?q={encoded_query}&src=typed_query&f=live"
    
    def _extract_tweet_data(self, tweet_element) -> Dict[str, Any]:
        """Extrair dados de um tweet do elemento HTML"""
        try:
            # Extrair texto do tweet
            text_element = tweet_element.find_element(By.CSS_SELECTOR, '[data-testid="tweetText"]')
            text = text_element.text if text_element else ""
            
            # Extrair nome de usuário
            username_element = tweet_element.find_element(By.CSS_SELECTOR, '[data-testid="User-Name"] a')
            username = username_element.get_attribute('href').split('/')[-1] if username_element else "unknown"
            
            # Extrair timestamp
            time_element = tweet_element.find_element(By.CSS_SELECTOR, 'time')
            timestamp_str = time_element.get_attribute('datetime') if time_element else None
            
            # Converter timestamp
            if timestamp_str:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            else:
                timestamp = datetime.now()
            
            # Extrair métricas (likes, retweets, etc.)
            likes = 0
            retweets = 0
            replies = 0
            
            try:
                # Buscar elementos de métricas
                metrics_elements = tweet_element.find_elements(By.CSS_SELECTOR, '[role="group"] [data-testid] span')
                for element in metrics_elements:
                    text_metric = element.text
                    if text_metric and text_metric.isdigit():
                        # Determinar tipo de métrica baseado no contexto
                        parent_testid = element.find_element(By.XPATH, '../..').get_attribute('data-testid')
                        if 'like' in parent_testid:
                            likes = int(text_metric)
                        elif 'retweet' in parent_testid:
                            retweets = int(text_metric)
                        elif 'reply' in parent_testid:
                            replies = int(text_metric)
            except:
                pass  # Métricas podem não estar disponíveis
            
            # Gerar ID único para o tweet
            tweet_id = hashlib.md5(f"{username}_{text}_{timestamp}".encode()).hexdigest()[:12]
            
            return {
                'platform': 'twitter',
                'id': tweet_id,
                'text': text,
                'timestamp': timestamp,
                'user_hash': self._hash_user_id(username),
                'username': username,
                'likes': likes,
                'retweets': retweets,
                'replies': replies,
                'url': f"https://twitter.com/{username}/status/{tweet_id}",
                'lang': 'pt',  # Assumir português baseado no filtro
                'collected_at': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Erro ao extrair dados do tweet: {e}")
            return None
    
    def scrape(self, keywords: List[str], limit: int = 800) -> List[Dict[str, Any]]:
        """
        Scrape tweets baseado nas keywords usando Selenium
        
        Args:
            keywords: Lista de palavras-chave para busca
            limit: Número máximo de tweets para coletar
            
        Returns:
            Lista de dicionários com dados dos tweets
        """
        search_url = self._build_search_url(keywords)
        logger.info(f"Scraping Twitter com URL: {search_url}")
        logger.info(f"Limite: {limit} tweets")
        
        tweets_data = []
        
        try:
            self._setup_driver()
            
            # Navegar para a página de busca
            self.driver.get(search_url)
            
            # Aguardar carregamento da página
            time.sleep(3)
            
            # Scroll e coleta de tweets
            last_height = self.driver.execute_script("return document.body.scrollHeight")
            no_new_tweets_count = 0
            
            while len(tweets_data) < limit and no_new_tweets_count < 3:
                # Buscar elementos de tweet
                tweet_elements = self.driver.find_elements(By.CSS_SELECTOR, '[data-testid="tweet"]')
                
                initial_count = len(tweets_data)
                
                for tweet_element in tweet_elements[len(tweets_data):]:
                    if len(tweets_data) >= limit:
                        break
                    
                    tweet_data = self._extract_tweet_data(tweet_element)
                    if tweet_data:
                        # Filtrar retweets se configurado
                        if self.twitter_config.get('exclude_retweets', True) and tweet_data['text'].startswith('RT @'):
                            continue
                        
                        tweets_data.append(tweet_data)
                        
                        # Log progresso a cada 10 tweets
                        if len(tweets_data) % 10 == 0:
                            logger.info(f"Coletados {len(tweets_data)} tweets")
                
                # Verificar se conseguiu novos tweets
                if len(tweets_data) == initial_count:
                    no_new_tweets_count += 1
                else:
                    no_new_tweets_count = 0
                
                # Scroll para carregar mais conteúdo
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
                
                # Verificar se a página cresceu
                new_height = self.driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    no_new_tweets_count += 1
                else:
                    last_height = new_height
            
        except Exception as e:
            logger.error(f"Erro ao scraping Twitter: {str(e)}")
            raise
        finally:
            self._close_driver()
        
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