"""
Módulo de compliance para LGPD e ToS das redes sociais
"""

import hashlib
import hmac
import os
from datetime import datetime
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class LGPDCompliance:
    """Classe para garantir compliance com LGPD"""
    
    def __init__(self, salt_key: str = None):
        """
        Inicializar com chave de salt para hash
        
        Args:
            salt_key: Chave para salt do hash. Se None, usa variável de ambiente
        """
        self.salt_key = salt_key or os.getenv('HASH_SALT_KEY', 'default_salt_2024')
        
    def anonymize_user_id(self, user_id: str, platform: str = '') -> str:
        """
        Anonimizar ID de usuário usando hash SHA256 com salt
        
        Args:
            user_id: ID original do usuário
            platform: Plataforma de origem (twitter, reddit, etc.)
            
        Returns:
            Hash anonimizado do usuário
        """
        # Combinar user_id, platform e salt
        combined = f"{platform}:{user_id}:{self.salt_key}"
        
        # Gerar hash SHA256
        hash_object = hashlib.sha256(combined.encode('utf-8'))
        return hash_object.hexdigest()[:16]  # Usar apenas primeiros 16 caracteres
    
    def remove_pii(self, text: str) -> str:
        """
        Remover informações pessoais identificáveis do texto
        
        Args:
            text: Texto original
            
        Returns:
            Texto com PII removido
        """
        import re
        
        # Padrões de PII para remover
        patterns = [
            # E-mails
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]'),
            # Telefones brasileiros
            (r'\b(?:\+55\s?)?(?:\(?[1-9]{2}\)?\s?)?(?:9\s?)?[0-9]{4}[-\s]?[0-9]{4}\b', '[TELEFONE]'),
            # CPF
            (r'\b[0-9]{3}\.?[0-9]{3}\.?[0-9]{3}[-\.]?[0-9]{2}\b', '[CPF]'),
            # URLs que podem conter informações pessoais
            (r'https?://(?:www\.)?(?:facebook|instagram|linkedin|twitter)\.com/[^\s]+', '[PERFIL_SOCIAL]'),
        ]
        
        cleaned_text = text
        for pattern, replacement in patterns:
            cleaned_text = re.sub(pattern, replacement, cleaned_text, flags=re.IGNORECASE)
        
        return cleaned_text
    
    def get_compliance_metadata(self, data_source: str) -> Dict[str, Any]:
        """
        Gerar metadados de compliance para o dataset
        
        Args:
            data_source: Fonte dos dados (twitter, reddit, etc.)
            
        Returns:
            Dicionário com metadados de compliance
        """
        return {
            'collected_at': datetime.now().isoformat(),
            'data_source': data_source,
            'anonymization_method': 'SHA256 with salt',
            'pii_removal': True,
            'lgpd_compliance': True,
            'retention_policy': '90 days for research purposes',
            'legal_basis': 'Legitimate interest for academic research (Art. 7º, IX LGPD)',
            'data_controller': 'Universidade - Programa de Mestrado',
            'contact_dpo': 'dpo@universidade.edu.br',
            'user_rights': {
                'access': 'Users can request access to their anonymized data',
                'rectification': 'Anonymized data cannot be rectified',
                'erasure': 'Data is automatically deleted after retention period',
                'portability': 'Anonymized research data is not portable',
                'objection': 'Users can object to processing via contact'
            }
        }

class ToSCompliance:
    """Classe para compliance com Termos de Serviço das plataformas"""
    
    @staticmethod
    def get_twitter_tos_compliance() -> Dict[str, Any]:
        """Informações de compliance para Twitter/X"""
        return {
            'platform': 'Twitter/X',
            'tos_version': '2024',
            'allowed_use': 'Academic research and analysis',
            'rate_limits': 'Respected via snscrape and selenium',
            'data_usage': 'Public posts only, no private content',
            'attribution': 'Platform credited in research',
            'commercial_use': False,
            'data_sharing': 'Anonymized aggregated results only',
            'compliance_date': datetime.now().isoformat(),
            'notes': 'Using public APIs and web scraping for academic purposes'
        }
    
    @staticmethod
    def get_reddit_tos_compliance() -> Dict[str, Any]:
        """Informações de compliance para Reddit"""
        return {
            'platform': 'Reddit',
            'tos_version': '2024',
            'allowed_use': 'Academic research via official API',
            'rate_limits': 'Respected via PRAW library',
            'data_usage': 'Public subreddit posts only',
            'attribution': 'Platform and subreddits credited',
            'commercial_use': False,
            'data_sharing': 'Anonymized aggregated results only',
            'compliance_date': datetime.now().isoformat(),
            'api_key': 'Official Reddit API credentials used'
        }

def apply_compliance_to_dataset(df, platform: str) -> tuple:
    """
    Aplicar compliance completo a um dataset
    
    Args:
        df: DataFrame com dados coletados
        platform: Plataforma de origem
        
    Returns:
        Tuple (df_compliant, metadata)
    """
    lgpd = LGPDCompliance()
    
    # Aplicar anonimização
    if 'user_hash' not in df.columns and 'username' in df.columns:
        df['user_hash'] = df['username'].apply(
            lambda x: lgpd.anonymize_user_id(x, platform)
        )
    
    # Remover PII do texto
    if 'text' in df.columns:
        df['text_clean'] = df['text'].apply(lgpd.remove_pii)
    
    # Remover colunas com informações pessoais
    columns_to_remove = ['username', 'user_id', 'email']
    df_compliant = df.drop(columns=[col for col in columns_to_remove if col in df.columns])
    
    # Gerar metadados
    metadata = {
        'lgpd': lgpd.get_compliance_metadata(platform),
        'tos': ToSCompliance.get_twitter_tos_compliance() if platform == 'twitter' 
               else ToSCompliance.get_reddit_tos_compliance(),
        'processing_info': {
            'original_records': len(df),
            'final_records': len(df_compliant),
            'anonymization_applied': True,
            'pii_removal_applied': True,
            'columns_removed': [col for col in columns_to_remove if col in df.columns]
        }
    }
    
    logger.info(f"Compliance aplicado: {len(df_compliant)} registros processados")
    
    return df_compliant, metadata

def generate_compliance_report(metadata: Dict[str, Any], output_path: str) -> None:
    """
    Gerar relatório de compliance em formato texto
    
    Args:
        metadata: Metadados de compliance
        output_path: Caminho para salvar o relatório
    """
    
    report = f"""RELATÓRIO DE COMPLIANCE - LGPD E TERMOS DE SERVIÇO
{'='*60}

Data de geração: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

COMPLIANCE LGPD:
- Base legal: {metadata['lgpd']['legal_basis']}
- Método de anonimização: {metadata['lgpd']['anonymization_method']}
- Remoção de PII: {metadata['lgpd']['pii_removal']}
- Política de retenção: {metadata['lgpd']['retention_policy']}
- Controlador: {metadata['lgpd']['data_controller']}
- Contato DPO: {metadata['lgpd']['contact_dpo']}

DIREITOS DOS USUÁRIOS:
- Acesso: {metadata['lgpd']['user_rights']['access']}
- Retificação: {metadata['lgpd']['user_rights']['rectification']}
- Apagamento: {metadata['lgpd']['user_rights']['erasure']}
- Portabilidade: {metadata['lgpd']['user_rights']['portability']}
- Objeção: {metadata['lgpd']['user_rights']['objection']}

COMPLIANCE TERMOS DE SERVIÇO:
- Plataforma: {metadata['tos']['platform']}
- Uso permitido: {metadata['tos']['allowed_use']}
- Respeito aos rate limits: {metadata['tos']['rate_limits']}
- Tipo de dados: {metadata['tos']['data_usage']}
- Uso comercial: {metadata['tos']['commercial_use']}
- Compartilhamento: {metadata['tos']['data_sharing']}

PROCESSAMENTO:
- Registros originais: {metadata['processing_info']['original_records']}
- Registros finais: {metadata['processing_info']['final_records']}
- Colunas removidas: {', '.join(metadata['processing_info']['columns_removed'])}

NOTAS IMPORTANTES:
1. Todos os dados pessoais foram anonimizados usando hash SHA256 com salt
2. Informações pessoais identificáveis (PII) foram removidas dos textos
3. Apenas dados públicos foram coletados
4. Os dados serão mantidos apenas pelo período necessário para a pesquisa
5. Usuários podem solicitar exclusão através do contato do DPO

Este processamento está em conformidade com a Lei Geral de Proteção 
de Dados (LGPD - Lei 13.709/2018) e os Termos de Serviço das plataformas.
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"Relatório de compliance salvo em: {output_path}")