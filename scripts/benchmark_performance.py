#!/usr/bin/env python3
"""
Script para fazer benchmark de inferência GPU vs CPU.
Compara performance entre dispositivos.
"""

import argparse
import time
import sys
import os
from pathlib import Path
import json

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

def benchmark_inference(device, num_samples=1000, batch_size=32):
    """Executa benchmark de inferência em um dispositivo"""
    
    print(f"🔄 Executando benchmark em {device.upper()}...")
    
    try:
        import torch
        import numpy as np
        
        # Configurar device
        if device == 'cuda' and not torch.cuda.is_available():
            print("❌ CUDA não disponível, usando CPU")
            device = 'cpu'
        
        torch_device = torch.device(device)
        
        # Simular modelo simples (transformador pequeno)
        vocab_size = 10000
        embed_dim = 256
        hidden_dim = 512
        num_classes = 3
        
        # Modelo simples para benchmark
        model = torch.nn.Sequential(
            torch.nn.Embedding(vocab_size, embed_dim),
            torch.nn.Linear(embed_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(), 
            torch.nn.Linear(hidden_dim, num_classes),
            torch.nn.Softmax(dim=1)
        ).to(torch_device)
        
        # Dados sintéticos para teste
        max_length = 128
        test_data = torch.randint(0, vocab_size, (num_samples, max_length)).to(torch_device)
        
        # Warm-up
        with torch.no_grad():
            _ = model(test_data[:batch_size])
        
        # Benchmark
        times = []
        model.eval()
        
        print(f"   📊 Processando {num_samples} amostras em batches de {batch_size}")
        
        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                end_idx = min(i + batch_size, num_samples)
                batch = test_data[i:end_idx]
                
                start_time = time.perf_counter()
                outputs = model(batch)
                
                # Sincronizar GPU se necessário
                if device == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                times.append(end_time - start_time)
        
        # Calcular estatísticas
        total_time = sum(times)
        avg_time_per_batch = total_time / len(times)
        avg_time_per_sample = total_time / num_samples
        throughput = num_samples / total_time
        
        results = {
            'device': device,
            'num_samples': num_samples,
            'batch_size': batch_size,
            'total_time_seconds': total_time,
            'avg_time_per_batch_ms': avg_time_per_batch * 1000,
            'avg_time_per_sample_ms': avg_time_per_sample * 1000,
            'throughput_samples_per_second': throughput,
            'model_parameters': sum(p.numel() for p in model.parameters())
        }
        
        return results
        
    except ImportError as e:
        print(f"❌ Erro de importação: {e}")
        return None
    except Exception as e:
        print(f"❌ Erro durante benchmark: {e}")
        return None

def benchmark_text_processing(num_texts=1000):
    """Benchmark de processamento de texto"""
    
    print(f"🔄 Benchmark de processamento de texto ({num_texts} textos)...")
    
    try:
        import re
        import string
        
        # Gerar textos sintéticos
        import random
        
        words = ['análise', 'sentimento', 'positivo', 'negativo', 'neutro', 
                'twitter', 'reddit', 'social', 'mídia', 'opinião',
                'produto', 'serviço', 'qualidade', 'atendimento', 'experiência']
        
        texts = []
        for _ in range(num_texts):
            text_length = random.randint(10, 200)
            text = ' '.join(random.choices(words, k=text_length))
            texts.append(text)
        
        # Função de limpeza simples
        def clean_text(text):
            # Remover URLs
            text = re.sub(r'http\S+|www\S+', '', text)
            # Remover menções
            text = re.sub(r'@\w+', '', text)
            # Remover pontuação
            text = text.translate(str.maketrans('', '', string.punctuation))
            # Lowercase
            text = text.lower()
            # Remover espaços extras
            text = ' '.join(text.split())
            return text
        
        # Benchmark
        start_time = time.perf_counter()
        
        cleaned_texts = [clean_text(text) for text in texts]
        
        end_time = time.perf_counter()
        
        processing_time = end_time - start_time
        
        results = {
            'num_texts': num_texts,
            'total_time_seconds': processing_time,
            'avg_time_per_text_ms': (processing_time / num_texts) * 1000,
            'throughput_texts_per_second': num_texts / processing_time
        }
        
        return results
        
    except Exception as e:
        print(f"❌ Erro no benchmark de texto: {e}")
        return None

def main():
    """Função principal"""
    parser = argparse.ArgumentParser(description='Benchmark de performance ML')
    parser.add_argument('--samples', type=int, default=1000,
                        help='Número de amostras para teste')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Tamanho do batch')
    parser.add_argument('--output', type=str, default='results/benchmark.json',
                        help='Arquivo para salvar resultados')
    parser.add_argument('--devices', nargs='+', default=['cpu', 'cuda'],
                        help='Dispositivos para testar')
    
    args = parser.parse_args()
    
    print("🚀 Benchmark de Performance - ML Pipeline")
    print("=" * 45)
    
    all_results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'inference_benchmarks': {},
        'text_processing_benchmark': {}
    }
    
    # Benchmark de inferência por device
    for device in args.devices:
        print(f"\n🎯 Testando device: {device}")
        results = benchmark_inference(device, args.samples, args.batch_size)
        
        if results:
            all_results['inference_benchmarks'][device] = results
            
            print(f"✅ Resultados {device.upper()}:")
            print(f"   ⏱️  Tempo total: {results['total_time_seconds']:.2f}s")
            print(f"   📊 Throughput: {results['throughput_samples_per_second']:.1f} samples/s")
            print(f"   🔍 Tempo/amostra: {results['avg_time_per_sample_ms']:.2f}ms")
    
    # Benchmark de processamento de texto
    print(f"\n📝 Benchmark de Processamento de Texto:")
    text_results = benchmark_text_processing(args.samples)
    
    if text_results:
        all_results['text_processing_benchmark'] = text_results
        
        print(f"✅ Resultados processamento:")
        print(f"   ⏱️  Tempo total: {text_results['total_time_seconds']:.3f}s")
        print(f"   📊 Throughput: {text_results['throughput_texts_per_second']:.1f} texts/s")
        print(f"   🔍 Tempo/texto: {text_results['avg_time_per_text_ms']:.3f}ms")
    
    # Comparação entre devices
    if len(all_results['inference_benchmarks']) > 1:
        print(f"\n⚡ Comparação de Performance:")
        
        devices = list(all_results['inference_benchmarks'].keys())
        if 'cpu' in devices and 'cuda' in devices:
            cpu_throughput = all_results['inference_benchmarks']['cpu']['throughput_samples_per_second']
            gpu_throughput = all_results['inference_benchmarks']['cuda']['throughput_samples_per_second']
            
            speedup = gpu_throughput / cpu_throughput
            print(f"   🚀 GPU Speedup: {speedup:.2f}x mais rápido que CPU")
            
            all_results['gpu_speedup'] = speedup
    
    # Salvar resultados
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Resultados salvos em: {args.output}")
    
    # Resumo final
    print(f"\n🎉 Benchmark Completo!")
    print(f"📊 Devices testados: {len(all_results['inference_benchmarks'])}")
    
    if 'gpu_speedup' in all_results:
        print(f"⚡ GPU é {all_results['gpu_speedup']:.1f}x mais rápido")

if __name__ == "__main__":
    main()
