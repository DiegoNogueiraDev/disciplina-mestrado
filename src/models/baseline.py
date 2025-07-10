import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import logging
from typing import Dict, List, Tuple, Any
import joblib

logger = logging.getLogger(__name__)

class BaselineClassifier:
    """Modelo baseline usando TF-IDF + Regressão Logística"""
    
    def __init__(self, max_features: int = 10000, ngram_range: Tuple[int, int] = (1, 2)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        
        # Inicializar componentes
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words=None,  # Não usar stop words em português automáticas
            lowercase=True,
            min_df=2,
            max_df=0.95
        )
        
        self.classifier = LogisticRegression(
            random_state=42,
            max_iter=1000,
            solver='liblinear',
            multi_class='ovr'
        )
        
        self.label_encoder = {'negative': 0, 'neutral': 1, 'positive': 2}
        self.label_decoder = {0: 'negative', 1: 'neutral', 2: 'positive'}
        
        self.is_fitted = False
    
    def prepare_data(self, df: pd.DataFrame, text_column: str = 'lemmatized', 
                    label_column: str = 'sentiment') -> Tuple[List[str], List[int]]:
        """Preparar dados para treinamento"""
        
        # Filtrar dados válidos
        valid_data = df.dropna(subset=[text_column, label_column])
        
        texts = valid_data[text_column].tolist()
        labels = [self.label_encoder[label] for label in valid_data[label_column]]
        
        logger.info(f"Dados preparados: {len(texts)} amostras")
        logger.info(f"Distribuição de classes: {pd.Series(labels).value_counts().to_dict()}")
        
        return texts, labels
    
    def fit(self, texts: List[str], labels: List[int]) -> None:
        """Treinar o modelo baseline"""
        logger.info("Iniciando treinamento do modelo baseline...")
        
        # Vetorizar textos
        logger.info("Vetorizando textos com TF-IDF...")
        X = self.vectorizer.fit_transform(texts)
        logger.info(f"Matriz TF-IDF: {X.shape}")
        
        # Treinar classificador
        logger.info("Treinando Regressão Logística...")
        self.classifier.fit(X, labels)
        
        self.is_fitted = True
        logger.info("Treinamento concluído!")
    
    def cross_validate(self, texts: List[str], labels: List[int], cv: int = 5) -> Dict[str, Any]:
        """Fazer validação cruzada 5-fold"""
        logger.info(f"Executando validação cruzada {cv}-fold...")
        
        # Vetorizar textos
        X = self.vectorizer.fit_transform(texts)
        
        # Validação cruzada estratificada
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        # Métricas
        scores = cross_val_score(self.classifier, X, labels, cv=skf, scoring='f1_macro')
        
        results = {
            'cv_scores': scores,
            'mean_f1': scores.mean(),
            'std_f1': scores.std(),
            'scores_detail': {
                'f1_macro': cross_val_score(self.classifier, X, labels, cv=skf, scoring='f1_macro'),
                'accuracy': cross_val_score(self.classifier, X, labels, cv=skf, scoring='accuracy'),
                'precision_macro': cross_val_score(self.classifier, X, labels, cv=skf, scoring='precision_macro'),
                'recall_macro': cross_val_score(self.classifier, X, labels, cv=skf, scoring='recall_macro')
            }
        }
        
        logger.info(f"F1-macro: {results['mean_f1']:.3f} (±{results['std_f1']:.3f})")
        return results
    
    def predict(self, texts: List[str]) -> List[str]:
        """Fazer predições"""
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")
        
        X = self.vectorizer.transform(texts)
        predictions = self.classifier.predict(X)
        
        return [self.label_decoder[pred] for pred in predictions]
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Obter probabilidades das predições"""
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")
        
        X = self.vectorizer.transform(texts)
        return self.classifier.predict_proba(X)
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, List[Tuple[str, float]]]:
        """Obter features mais importantes por classe"""
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")
        
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Coeficientes por classe
        importance_by_class = {}
        
        for class_idx, class_name in self.label_decoder.items():
            if len(self.classifier.classes_) > 2:
                coefs = self.classifier.coef_[class_idx]
            else:
                coefs = self.classifier.coef_[0] if class_idx == 1 else -self.classifier.coef_[0]
            
            # Top features positivas e negativas
            top_positive_idx = np.argsort(coefs)[-top_n:][::-1]
            top_negative_idx = np.argsort(coefs)[:top_n]
            
            importance_by_class[class_name] = {
                'positive': [(feature_names[idx], coefs[idx]) for idx in top_positive_idx],
                'negative': [(feature_names[idx], coefs[idx]) for idx in top_negative_idx]
            }
        
        return importance_by_class
    
    def evaluate(self, texts: List[str], true_labels: List[int]) -> Dict[str, Any]:
        """Avaliar modelo em conjunto de teste"""
        predictions = self.predict(texts)
        pred_labels = [self.label_encoder[pred] for pred in predictions]
        
        # Relatório de classificação
        report = classification_report(
            true_labels, pred_labels, 
            target_names=list(self.label_decoder.values()),
            output_dict=True
        )
        
        # Matriz de confusão
        cm = confusion_matrix(true_labels, pred_labels)
        
        return {
            'classification_report': report,
            'confusion_matrix': cm,
            'accuracy': report['accuracy'],
            'macro_f1': report['macro avg']['f1-score'],
            'weighted_f1': report['weighted avg']['f1-score']
        }
    
    def save_model(self, filepath: str) -> None:
        """Salvar modelo treinado"""
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")
        
        model_data = {
            'vectorizer': self.vectorizer,
            'classifier': self.classifier,
            'label_encoder': self.label_encoder,
            'label_decoder': self.label_decoder,
            'max_features': self.max_features,
            'ngram_range': self.ngram_range
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Modelo salvo em: {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Carregar modelo treinado"""
        model_data = joblib.load(filepath)
        
        self.vectorizer = model_data['vectorizer']
        self.classifier = model_data['classifier']
        self.label_encoder = model_data['label_encoder']
        self.label_decoder = model_data['label_decoder']
        self.max_features = model_data['max_features']
        self.ngram_range = model_data['ngram_range']
        
        self.is_fitted = True
        logger.info(f"Modelo carregado de: {filepath}")

def create_sample_labeled_data(df: pd.DataFrame, sample_size: int = 600) -> pd.DataFrame:
    """Criar dados de exemplo com rótulos simulados para demonstração"""
    
    # Selecionar amostra
    sample_df = df.sample(n=min(sample_size, len(df)), random_state=42).copy()
    
    # Simular rotulagem baseada em palavras-chave (para demonstração)
    def simulate_label(text: str) -> str:
        text_lower = text.lower()
        
        # Palavras indicativas de sentimento
        positive_words = ['bom', 'ótimo', 'excelente', 'positivo', 'benefício', 'melhora', 'sucesso']
        negative_words = ['ruim', 'péssimo', 'horrível', 'negativo', 'problema', 'erro', 'fracasso', 'preocupado']
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            return 'positive'
        elif neg_count > pos_count:
            return 'negative'
        else:
            # Distribuição mais realista: mais neutros
            return np.random.choice(['neutral', 'positive', 'negative'], p=[0.6, 0.2, 0.2])
    
    sample_df['sentiment'] = sample_df['lemmatized'].apply(simulate_label)
    
    return sample_df