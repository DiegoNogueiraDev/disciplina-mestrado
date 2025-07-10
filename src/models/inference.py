import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
import logging
import pickle
from pathlib import Path
import fasttext
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import tempfile
import os

logger = logging.getLogger(__name__)

class FastTextMLPModel(nn.Module):
    """Modelo MLP para classificação de sentimento usando embeddings FastText"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 3, dropout: float = 0.3):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        return self.classifier(x)

class EarlyStopping:
    """Early stopping para evitar overfitting"""
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

class SentimentTrainer:
    """Classe para treinar modelo FastText + MLP com early stopping"""
    
    def __init__(self, embedding_dim: int = 300, hidden_dim: int = 128, 
                 output_dim: int = 3, dropout: float = 0.3):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        
        self.model = None
        self.fasttext_model = None
        self.label_encoder = {'negative': 0, 'neutral': 1, 'positive': 2}
        self.label_decoder = {0: 'negative', 1: 'neutral', 2: 'positive'}
        
        # Histórico de treinamento
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        logger.info(f"Treinamento configurado para: {self.device}")
    
    def train_fasttext_embeddings(self, texts: List[str], model_path: str = None) -> str:
        """Treinar embeddings FastText"""
        logger.info("Treinando embeddings FastText...")
        
        # Criar arquivo temporário com textos
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
        try:
            for text in texts:
                temp_file.write(text + '\n')
            temp_file.close()
            
            # Treinar modelo FastText
            model = fasttext.train_unsupervised(
                temp_file.name,
                model='skipgram',
                dim=self.embedding_dim,
                epoch=20,
                lr=0.05,
                wordNgrams=2,
                minCount=2,
                thread=4
            )
            
            # Salvar modelo
            if model_path is None:
                model_path = 'models/fasttext_embeddings.bin'
            
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            model.save_model(model_path)
            
            logger.info(f"Embeddings FastText salvos em: {model_path}")
            return model_path
            
        finally:
            # Limpar arquivo temporário
            os.unlink(temp_file.name)
    
    def load_fasttext_model(self, model_path: str) -> None:
        """Carregar modelo FastText"""
        self.fasttext_model = fasttext.load_model(model_path)
        logger.info(f"Modelo FastText carregado: {model_path}")
    
    def prepare_data(self, df: pd.DataFrame, text_column: str = 'lemmatized', 
                    label_column: str = 'sentiment', test_size: float = 0.2) -> Tuple:
        """Preparar dados para treinamento"""
        
        # Filtrar dados válidos
        valid_data = df.dropna(subset=[text_column, label_column])
        
        texts = valid_data[text_column].tolist()
        labels = [self.label_encoder[label] for label in valid_data[label_column]]
        
        # Dividir em treino/validação
        X_train, X_val, y_train, y_val = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        logger.info(f"Dados preparados:")
        logger.info(f"  Treino: {len(X_train)} amostras")
        logger.info(f"  Validação: {len(X_val)} amostras")
        logger.info(f"  Classes: {len(set(labels))}")
        
        return X_train, X_val, y_train, y_val
    
    def texts_to_embeddings(self, texts: List[str]) -> np.ndarray:
        """Converter textos em embeddings usando FastText"""
        if self.fasttext_model is None:
            raise ValueError("Modelo FastText não carregado")
        
        embeddings = []
        for text in texts:
            embedding = self.fasttext_model.get_sentence_vector(text)
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def create_data_loader(self, texts: List[str], labels: List[int], 
                          batch_size: int = 32) -> DataLoader:
        """Criar DataLoader para PyTorch"""
        
        # Converter para embeddings
        embeddings = self.texts_to_embeddings(texts)
        
        # Converter para tensors
        X_tensor = torch.FloatTensor(embeddings)
        y_tensor = torch.LongTensor(labels)
        
        # Criar dataset e dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        return dataloader
    
    def train(self, X_train: List[str], y_train: List[int], 
              X_val: List[str], y_val: List[int],
              epochs: int = 50, batch_size: int = 32, 
              learning_rate: float = 0.001, patience: int = 10) -> Dict[str, Any]:
        """Treinar modelo com early stopping"""
        
        logger.info("Iniciando treinamento do modelo...")
        
        # Criar modelo
        self.model = FastTextMLPModel(
            input_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            dropout=self.dropout
        ).to(self.device)
        
        # Criar data loaders
        train_loader = self.create_data_loader(X_train, y_train, batch_size)
        val_loader = self.create_data_loader(X_val, y_val, batch_size)
        
        # Configurar otimizador e loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        
        # Early stopping
        early_stopping = EarlyStopping(patience=patience)
        
        # Loop de treinamento
        for epoch in range(epochs):
            # Treino
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            # Validação
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            # Calcular métricas
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            
            # Salvar histórico
            self.train_losses.append(avg_train_loss)
            self.val_losses.append(avg_val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Scheduler
            scheduler.step(avg_val_loss)
            
            # Log
            if epoch % 5 == 0:
                logger.info(f"Epoch {epoch:3d}: Train Loss: {avg_train_loss:.4f}, "
                          f"Val Loss: {avg_val_loss:.4f}, "
                          f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
            
            # Early stopping
            if early_stopping(avg_val_loss, self.model):
                logger.info(f"Early stopping na época {epoch}")
                break
        
        logger.info("Treinamento concluído!")
        
        return {
            'final_epoch': epoch,
            'best_val_loss': early_stopping.best_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }
    
    def plot_training_history(self, save_path: str = None) -> None:
        """Plotar curvas de treinamento"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        epochs = range(1, len(self.train_losses) + 1)
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy
        ax2.plot(epochs, self.train_accuracies, 'b-', label='Training Accuracy')
        ax2.plot(epochs, self.val_accuracies, 'r-', label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Gráfico de treinamento salvo em: {save_path}")
        
        plt.show()
    
    def evaluate(self, X_test: List[str], y_test: List[int]) -> Dict[str, Any]:
        """Avaliar modelo no conjunto de teste"""
        if self.model is None:
            raise ValueError("Modelo não foi treinado")
        
        self.model.eval()
        
        # Fazer predições
        embeddings = self.texts_to_embeddings(X_test)
        X_tensor = torch.FloatTensor(embeddings).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
            probabilities = torch.softmax(outputs, dim=1)
        
        predictions = predicted.cpu().numpy()
        probs = probabilities.cpu().numpy()
        
        # Relatório de classificação
        report = classification_report(
            y_test, predictions,
            target_names=list(self.label_decoder.values()),
            output_dict=True
        )
        
        # Matriz de confusão
        cm = confusion_matrix(y_test, predictions)
        
        return {
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': predictions,
            'probabilities': probs,
            'accuracy': report['accuracy'],
            'macro_f1': report['macro avg']['f1-score'],
            'weighted_f1': report['weighted avg']['f1-score']
        }
    
    def save_model(self, model_path: str, fasttext_path: str = None) -> None:
        """Salvar modelo treinado"""
        if self.model is None:
            raise ValueError("Modelo não foi treinado")
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'dropout': self.dropout,
            'label_encoder': self.label_encoder,
            'label_decoder': self.label_decoder,
            'train_history': {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'train_accuracies': self.train_accuracies,
                'val_accuracies': self.val_accuracies
            }
        }
        
        torch.save(checkpoint, model_path)
        logger.info(f"Modelo salvo em: {model_path}")

class SentimentInference:
    """Classe para inferência de sentimento usando modelo treinado"""
    
    def __init__(self, model_path: str = None, fasttext_model_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.fasttext_model = None
        self.label_encoder = None
        
        # Mapeamento de classes
        self.sentiment_labels = {0: 'negative', 1: 'neutral', 2: 'positive'}
        
        if model_path:
            self.load_model(model_path)
        if fasttext_model_path:
            self.load_fasttext_model(fasttext_model_path)
    
    def load_fasttext_model(self, model_path: str) -> None:
        """Carregar modelo FastText"""
        try:
            self.fasttext_model = fasttext.load_model(model_path)
            logger.info(f"Modelo FastText carregado: {model_path}")
        except Exception as e:
            logger.error(f"Erro ao carregar modelo FastText: {e}")
            raise
    
    def load_model(self, model_path: str) -> None:
        """Carregar modelo treinado"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Extrair dimensões do modelo
            input_dim = checkpoint['model_state_dict']['classifier.0.weight'].shape[1]
            
            # Criar modelo
            self.model = FastTextMLPModel(input_dim=input_dim)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # Carregar label encoder se disponível
            if 'label_encoder' in checkpoint:
                self.label_encoder = checkpoint['label_encoder']
            
            logger.info(f"Modelo carregado: {model_path}")
            logger.info(f"Usando dispositivo: {self.device}")
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            raise
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        """Obter embedding do texto usando FastText"""
        if self.fasttext_model is None:
            raise ValueError("Modelo FastText não carregado")
        
        try:
            # Obter embedding da sentença
            embedding = self.fasttext_model.get_sentence_vector(text)
            return embedding
        except Exception as e:
            logger.error(f"Erro ao obter embedding: {e}")
            return np.zeros(self.fasttext_model.get_dimension())
    
    def predict_single(self, text: str) -> Dict[str, Any]:
        """
        Predizer sentimento de um texto
        
        Args:
            text: Texto a ser analisado
            
        Returns:
            Dicionário com sentimento e confiança
        """
        if self.model is None:
            raise ValueError("Modelo não carregado")
        
        # Obter embedding
        embedding = self.get_text_embedding(text)
        
        # Converter para tensor
        input_tensor = torch.FloatTensor(embedding).unsqueeze(0).to(self.device)
        
        # Fazer predição
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = outputs.cpu().numpy()[0]
            predicted_class = np.argmax(probabilities)
            confidence = probabilities[predicted_class]
        
        return {
            'sentiment': self.sentiment_labels[predicted_class],
            'confidence': float(confidence),
            'probabilities': {
                'negative': float(probabilities[0]),
                'neutral': float(probabilities[1]),
                'positive': float(probabilities[2])
            }
        }
    
    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, Any]]:
        """
        Predizer sentimento em lote
        
        Args:
            texts: Lista de textos
            batch_size: Tamanho do lote
            
        Returns:
            Lista de predições
        """
        if self.model is None:
            raise ValueError("Modelo não carregado")
        
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = []
            
            # Obter embeddings do lote
            for text in batch_texts:
                embedding = self.get_text_embedding(text)
                batch_embeddings.append(embedding)
            
            # Converter para tensor
            batch_tensor = torch.FloatTensor(np.array(batch_embeddings)).to(self.device)
            
            # Fazer predições
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                probabilities = outputs.cpu().numpy()
                
                for j, probs in enumerate(probabilities):
                    predicted_class = np.argmax(probs)
                    confidence = probs[predicted_class]
                    
                    result = {
                        'text': batch_texts[j],
                        'sentiment': self.sentiment_labels[predicted_class],
                        'confidence': float(confidence),
                        'probabilities': {
                            'negative': float(probs[0]),
                            'neutral': float(probs[1]),
                            'positive': float(probs[2])
                        }
                    }
                    results.append(result)
            
            # Log progresso
            if (i + batch_size) % 100 == 0:
                logger.info(f"Processados {min(i + batch_size, len(texts))} de {len(texts)} textos")
        
        return results
    
    def predict_dataframe(self, df: pd.DataFrame, text_column: str = 'lemmatized') -> pd.DataFrame:
        """
        Predizer sentimento em DataFrame
        
        Args:
            df: DataFrame com textos
            text_column: Nome da coluna com texto
            
        Returns:
            DataFrame com predições adicionadas
        """
        logger.info(f"Iniciando predição para {len(df)} textos...")
        
        # Fazer predições
        predictions = self.predict_batch(df[text_column].tolist())
        
        # Criar DataFrame com resultados
        pred_df = pd.DataFrame(predictions)
        
        # Adicionar colunas de predição ao DataFrame original
        df_result = df.copy()
        df_result['predicted_sentiment'] = pred_df['sentiment']
        df_result['confidence'] = pred_df['confidence']
        df_result['prob_negative'] = pred_df['probabilities'].apply(lambda x: x['negative'])
        df_result['prob_neutral'] = pred_df['probabilities'].apply(lambda x: x['neutral'])
        df_result['prob_positive'] = pred_df['probabilities'].apply(lambda x: x['positive'])
        
        logger.info("Predição concluída")
        return df_result
    
    def get_prediction_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Obter estatísticas das predições"""
        if 'predicted_sentiment' not in df.columns:
            return {}
        
        sentiment_counts = df['predicted_sentiment'].value_counts()
        confidence_stats = df['confidence'].describe()
        
        return {
            'total_predictions': len(df),
            'sentiment_distribution': sentiment_counts.to_dict(),
            'sentiment_percentages': (sentiment_counts / len(df) * 100).to_dict(),
            'confidence_stats': confidence_stats.to_dict(),
            'avg_confidence': df['confidence'].mean(),
            'high_confidence_predictions': len(df[df['confidence'] > 0.8]),
            'low_confidence_predictions': len(df[df['confidence'] < 0.5])
        }