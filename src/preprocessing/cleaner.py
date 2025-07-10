import spacy
import re
import pandas as pd
from typing import List, Dict, Any
import logging
from langdetect import detect

logger = logging.getLogger(__name__)

class TextCleaner:
    """Classe para limpeza e pré-processamento de texto usando SpaCy PT-BR"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.filters = config.get('filters', {})
        
        # Carregar modelo SpaCy PT-BR
        try:
            self.nlp = spacy.load("pt_core_news_sm")
        except OSError:
            logger.error("Modelo SpaCy pt_core_news_sm não encontrado. Instale com: python -m spacy download pt_core_news_sm")
            raise
    
    def _remove_urls(self, text: str) -> str:
        """Remove URLs do texto"""
        url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        return url_pattern.sub('', text)
    
    def _remove_mentions(self, text: str) -> str:
        """Remove mentions (@usuario) do texto"""
        return re.sub(r'@\w+', '', text)
    
    def _remove_hashtags(self, text: str) -> str:
        """Remove hashtags (#hashtag) do texto"""
        return re.sub(r'#\w+', '', text)
    
    def _remove_extra_whitespace(self, text: str) -> str:
        """Remove espaços em branco extras"""
        return re.sub(r'\s+', ' ', text).strip()
    
    def _is_valid_length(self, text: str) -> bool:
        """Verifica se o texto tem tamanho válido"""
        min_length = self.filters.get('min_length', 10)
        max_length = self.filters.get('max_length', 300)
        return min_length <= len(text) <= max_length
    
    def _is_portuguese(self, text: str) -> bool:
        """Detectar se texto está em português"""
        try:
            return detect(text) == 'pt'
        except:
            return False
    
    def clean_text(self, text: str) -> str:
        """
        Limpar texto individual
        
        Args:
            text: Texto a ser limpo
            
        Returns:
            Texto limpo
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Normalização básica
        text = text.lower()
        
        # Remover URLs, mentions e hashtags
        text = self._remove_urls(text)
        text = self._remove_mentions(text)
        text = self._remove_hashtags(text)
        
        # Remover caracteres especiais (manter apenas letras, números e espaços)
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remover espaços extras
        text = self._remove_extra_whitespace(text)
        
        return text
    
    def lemmatize_text(self, text: str) -> str:
        """
        Lematizar texto usando SpaCy
        
        Args:
            text: Texto a ser lematizado
            
        Returns:
            Texto lematizado
        """
        if not text:
            return ""
        
        try:
            doc = self.nlp(text)
            # Manter apenas tokens que são palavras (não pontuação ou espaços)
            lemmatized = [token.lemma_ for token in doc if not token.is_punct and not token.is_space]
            return ' '.join(lemmatized)
        except:
            return text
    
    def process_text(self, text: str) -> Dict[str, Any]:
        """
        Processar texto completo (limpeza + lematização)
        
        Args:
            text: Texto original
            
        Returns:
            Dicionário com texto original, limpo e lematizado
        """
        # Limpeza inicial
        cleaned = self.clean_text(text)
        
        # Verificar se deve ser mantido
        is_valid = (
            self._is_valid_length(cleaned) and
            self._is_portuguese(cleaned) if self.filters.get('language') == 'pt' else True
        )
        
        # Lematização
        lemmatized = self.lemmatize_text(cleaned) if is_valid else ""
        
        return {
            'original': text,
            'cleaned': cleaned,
            'lemmatized': lemmatized,
            'is_valid': is_valid,
            'length': len(cleaned)
        }
    
    def process_dataframe(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """
        Processar DataFrame completo
        
        Args:
            df: DataFrame com dados
            text_column: Nome da coluna com texto
            
        Returns:
            DataFrame com colunas adicionais de texto processado
        """
        logger.info(f"Processando {len(df)} textos...")
        
        # Aplicar processamento
        processed_texts = df[text_column].apply(self.process_text)
        
        # Expandir resultados em colunas
        df_processed = pd.DataFrame(processed_texts.tolist())
        
        # Combinar com DataFrame original
        result_df = pd.concat([df, df_processed], axis=1)
        
        # Filtrar apenas textos válidos
        valid_df = result_df[result_df['is_valid']].copy()
        
        logger.info(f"Processamento concluído. {len(valid_df)} textos válidos de {len(df)} originais")
        
        return valid_df
    
    def get_processing_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Obter estatísticas do processamento"""
        if df.empty:
            return {}
        
        return {
            'total_texts': len(df),
            'valid_texts': sum(df['is_valid']) if 'is_valid' in df.columns else 0,
            'avg_length_original': df['original'].str.len().mean() if 'original' in df.columns else 0,
            'avg_length_cleaned': df['cleaned'].str.len().mean() if 'cleaned' in df.columns else 0,
            'avg_length_lemmatized': df['lemmatized'].str.len().mean() if 'lemmatized' in df.columns else 0,
            'languages_detected': df['original'].apply(self._is_portuguese).value_counts().to_dict() if 'original' in df.columns else {}
        }