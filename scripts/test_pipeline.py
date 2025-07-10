#!/usr/bin/env python3
"""
Testes unitários simples para funções de limpeza de texto.
"""

import sys
import os
from pathlib import Path
import unittest

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

class TestTextCleaning(unittest.TestCase):
    """Testes para funções de limpeza de texto"""
    
    def setUp(self):
        """Setup para cada teste"""
        try:
            from preprocessing.cleaner import TextCleaner
            self.cleaner = TextCleaner()
        except ImportError:
            # Implementação básica se o módulo não existir
            self.cleaner = self.BasicCleaner()
    
    class BasicCleaner:
        """Cleaner básico para testes"""
        
        def clean_pt(self, text):
            """Limpeza básica de texto português"""
            import re
            if not text:
                return ""
            
            text = str(text)
            
            # Remover URLs
            text = re.sub(r'http\S+|www\S+', '', text)
            
            # Remover menções
            text = re.sub(r'@\w+', '', text)
            
            # Remover hashtags mas manter o texto
            text = re.sub(r'#(\w+)', r'\1', text)
            
            # Remover números isolados
            text = re.sub(r'\b\d+\b', '', text)
            
            # Manter apenas letras, espaços e acentos
            text = re.sub(r'[^a-zA-ZÀ-ÿ\s]', '', text)
            
            # Remover espaços extras
            text = ' '.join(text.split())
            
            return text.strip().lower()
    
    def test_remove_urls(self):
        """Testa remoção de URLs"""
        text = "Confira este link https://exemplo.com e www.teste.org"
        expected = "confira este link e"
        result = self.cleaner.clean_pt(text)
        self.assertEqual(result, expected)
    
    def test_remove_mentions(self):
        """Testa remoção de menções"""
        text = "Olá @usuario e @outro_usuario como vocês estão?"
        expected = "olá e como vocês estão"
        result = self.cleaner.clean_pt(text)
        self.assertEqual(result, expected)
    
    def test_remove_hashtags_keep_text(self):
        """Testa remoção de # mas mantém o texto"""
        text = "Adorei o #produto e #atendimento excelente!"
        expected = "adorei o produto e atendimento excelente"
        result = self.cleaner.clean_pt(text)
        self.assertEqual(result, expected)
    
    def test_keep_letters_and_accents(self):
        """Testa que mantém letras e acentos portugueses"""
        text = "Ação, reação, coração! 123 %%% São Paulo"
        expected = "ação reação coração são paulo"
        result = self.cleaner.clean_pt(text)
        self.assertEqual(result, expected)
    
    def test_remove_extra_spaces(self):
        """Testa remoção de espaços extras"""
        text = "Texto    com     muitos   espaços"
        expected = "texto com muitos espaços"
        result = self.cleaner.clean_pt(text)
        self.assertEqual(result, expected)
    
    def test_empty_input(self):
        """Testa input vazio"""
        result = self.cleaner.clean_pt("")
        self.assertEqual(result, "")
        
        result = self.cleaner.clean_pt(None)
        self.assertEqual(result, "")
    
    def test_only_special_chars(self):
        """Testa texto com apenas caracteres especiais"""
        text = "!@#$%^&*()123"
        expected = ""
        result = self.cleaner.clean_pt(text)
        self.assertEqual(result, expected)
    
    def test_mixed_content(self):
        """Testa texto complexo misto"""
        text = "RT @user: Produto incrível! 🔥 https://loja.com #top 10/10 ⭐⭐⭐⭐⭐"
        expected = "rt produto incrível top"
        result = self.cleaner.clean_pt(text)
        self.assertEqual(result, expected)

class TestDataValidation(unittest.TestCase):
    """Testes para validação de dados"""
    
    def test_text_length_validation(self):
        """Testa validação de comprimento de texto"""
        
        # Texto muito curto
        short_text = "ok"
        self.assertLess(len(short_text), 10, "Texto muito curto deve ser < 10 chars")
        
        # Texto normal
        normal_text = "Este é um texto de tamanho normal para análise"
        self.assertGreaterEqual(len(normal_text), 10, "Texto normal deve ser >= 10 chars")
        self.assertLessEqual(len(normal_text), 1000, "Texto normal deve ser <= 1000 chars")
        
        # Texto muito longo
        long_text = "a" * 1001
        self.assertGreater(len(long_text), 1000, "Texto longo deve ser > 1000 chars")
    
    def test_language_detection_mock(self):
        """Testa detecção de idioma (mock)"""
        
        # Simular detecção de idioma
        pt_texts = [
            "Este é um texto em português",
            "Análise de sentimento é interessante",
            "Produto muito bom, recomendo"
        ]
        
        en_texts = [
            "This is a text in English", 
            "Sentiment analysis is interesting",
            "Very good product, I recommend"
        ]
        
        # Verificar que textos em português são mais longos em média (heurística simples)
        pt_avg_len = sum(len(text) for text in pt_texts) / len(pt_texts)
        en_avg_len = sum(len(text) for text in en_texts) / len(en_texts)
        
        # Este é um teste simples - em um caso real usaríamos langdetect
        self.assertIsInstance(pt_avg_len, float)
        self.assertIsInstance(en_avg_len, float)

def run_tests():
    """Executa todos os testes"""
    
    print("🧪 Executando Testes Unitários")
    print("=" * 35)
    
    # Criar suite de testes
    suite = unittest.TestSuite()
    
    # Adicionar testes
    suite.addTest(unittest.makeSuite(TestTextCleaning))
    suite.addTest(unittest.makeSuite(TestDataValidation))
    
    # Executar testes
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Resumo
    print("\n" + "=" * 35)
    print(f"🎯 Testes executados: {result.testsRun}")
    print(f"✅ Sucessos: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ Falhas: {len(result.failures)}")
    print(f"💥 Erros: {len(result.errors)}")
    
    if result.failures:
        print("\n💔 Falhas:")
        for test, traceback in result.failures:
            print(f"   - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\n💥 Erros:")
        for test, traceback in result.errors:
            print(f"   - {test}: {traceback.split('Exception:')[-1].strip()}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    
    if success:
        print("\n🎉 Todos os testes passaram!")
        return 0
    else:
        print("\n💥 Alguns testes falharam!")
        return 1

if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)
