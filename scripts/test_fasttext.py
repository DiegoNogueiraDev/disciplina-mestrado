#!/usr/bin/env python3
"""
Script para testar carregamento de embeddings FastText.
Verifica se embeddings carregam corretamente (GPU/CPU RAM).
"""

import os
import sys
import time
from pathlib import Path

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

def test_fasttext_model():
    """Testa carregamento de modelo FastText"""
    
    print("🔄 Testando importação FastText...")
    
    try:
        import fasttext
        print("✅ FastText importado com sucesso")
    except ImportError as e:
        print(f"❌ ERRO: FastText não instalado - {e}")
        print("🔧 Instale com: pip install fasttext")
        return False
    
    # Tentar baixar modelo português se não existir
    model_path = Path("models/lid.176.bin")
    
    if not model_path.exists():
        print("🔄 Modelo FastText não encontrado, tentando baixar...")
        try:
            # Criar diretório models se não existir
            model_path.parent.mkdir(exist_ok=True)
            
            print("📥 Baixando modelo de detecção de idioma (lid.176.bin)...")
            model = fasttext.load_model('https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin')
            
            # Salvar modelo localmente
            model.save_model(str(model_path))
            print(f"💾 Modelo salvo em: {model_path}")
            
        except Exception as e:
            print(f"❌ ERRO ao baixar modelo: {e}")
            print("🔧 Baixe manualmente de: https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin")
            return False
    
    try:
        print(f"🔄 Carregando modelo: {model_path}")
        start_time = time.time()
        
        # Carregar modelo
        model = fasttext.load_model(str(model_path))
        load_time = time.time() - start_time
        
        print(f"✅ Modelo carregado em {load_time:.2f}s")
        
        # Testar predição
        test_texts = [
            "Este é um texto em português",
            "This is a text in English",
            "Esto es un texto en español"
        ]
        
        print("🧪 Testando predições:")
        for text in test_texts:
            prediction = model.predict(text, k=1)
            lang = prediction[0][0].replace('__label__', '')
            confidence = prediction[1][0]
            print(f"   📝 '{text[:30]}...' → {lang} ({confidence:.3f})")
        
        # Informações do modelo
        print(f"\n📊 Informações do modelo:")
        print(f"   🏷️  Labels: {len(model.get_labels())}")
        print(f"   📐 Dimensão: {model.get_dimension()}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERRO ao carregar modelo: {e}")
        return False

def test_torch_availability():
    """Testa disponibilidade do PyTorch e CUDA"""
    
    print("\n🔄 Testando PyTorch...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} disponível")
        
        # Testar CUDA
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            
            print(f"🚀 CUDA disponível:")
            print(f"   🎮 Dispositivos: {device_count}")
            print(f"   🔧 Dispositivo atual: {current_device}")
            print(f"   📛 Nome: {device_name}")
            
            # Teste simples de tensor
            tensor = torch.randn(100, 100).cuda()
            result = torch.mm(tensor, tensor.t())
            print(f"   ✅ Teste de tensor: {result.shape}")
            
        else:
            print("⚠️  CUDA não disponível - usando CPU")
            
        return True
        
    except ImportError as e:
        print(f"❌ ERRO: PyTorch não instalado - {e}")
        return False

if __name__ == "__main__":
    print("🧪 Teste de Embeddings e ML Stack")
    print("=" * 40)
    
    # Testar FastText
    fasttext_ok = test_fasttext_model()
    
    # Testar PyTorch
    torch_ok = test_torch_availability()
    
    print("\n" + "=" * 40)
    if fasttext_ok and torch_ok:
        print("🎉 Todos os testes passaram!")
        sys.exit(0)
    else:
        print("💥 Alguns testes falharam")
        sys.exit(1)
