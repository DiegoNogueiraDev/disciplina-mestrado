import yaml
import os
from typing import Dict, Any
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

def load_config(config_path: str = "config/topic.yaml") -> Dict[str, Any]:
    """
    Carregar configuração do arquivo YAML
    
    Args:
        config_path: Caminho para o arquivo de configuração
        
    Returns:
        Dicionário com configurações
    """
    # Carregar variáveis de ambiente
    load_dotenv()
    
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        
        # Substituir variáveis de ambiente
        config = _substitute_env_vars(config)
        
        logger.info(f"Configuração carregada de: {config_path}")
        return config
        
    except FileNotFoundError:
        logger.error(f"Arquivo de configuração não encontrado: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Erro ao parsear YAML: {e}")
        raise

def _substitute_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Substituir variáveis de ambiente na configuração
    
    Args:
        config: Dicionário de configuração
        
    Returns:
        Configuração com variáveis substituídas
    """
    def substitute_value(value):
        if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
            env_var = value[2:-1]
            return os.getenv(env_var, value)
        elif isinstance(value, dict):
            return {k: substitute_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [substitute_value(item) for item in value]
        else:
            return value
    
    return substitute_value(config)

def setup_logging(level: str = "INFO") -> None:
    """
    Configurar logging do sistema
    
    Args:
        level: Nível de logging (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/sentiment_pipeline.log'),
            logging.StreamHandler()
        ]
    )
    
    # Criar diretório de logs se não existir
    os.makedirs('logs', exist_ok=True)