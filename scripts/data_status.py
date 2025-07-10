#!/usr/bin/env python3
"""
Script para verificar status dos dados no pipeline.
Mostra estatísticas dos dados em cada etapa do processo.
"""

import os
import sys
from pathlib import Path
import json

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

def check_data_raw():
    """Verifica dados brutos coletados"""
    
    raw_path = Path("data/raw")
    
    if not raw_path.exists():
        print("❌ Diretório data/raw não encontrado")
        return False
    
    files = list(raw_path.glob("*.json")) + list(raw_path.glob("*.csv"))
    
    if not files:
        print("⚠️  Nenhum arquivo encontrado em data/raw")
        return False
    
    print(f"📁 Data Raw ({len(files)} arquivos):")
    
    total_size = 0
    for file in files:
        size_mb = file.stat().st_size / (1024 * 1024)
        total_size += size_mb
        print(f"   📄 {file.name}: {size_mb:.2f} MB")
    
    print(f"   📊 Total: {total_size:.2f} MB")
    return True

def check_data_processed():
    """Verifica dados processados"""
    
    processed_path = Path("data/processed")
    
    if not processed_path.exists():
        print("❌ Diretório data/processed não encontrado")
        return False
    
    files = list(processed_path.glob("*.parquet")) + list(processed_path.glob("*.csv"))
    
    if not files:
        print("⚠️  Nenhum arquivo encontrado em data/processed")
        return False
    
    print(f"📁 Data Processed ({len(files)} arquivos):")
    
    total_size = 0
    for file in files:
        size_mb = file.stat().st_size / (1024 * 1024)
        total_size += size_mb
        print(f"   📄 {file.name}: {size_mb:.2f} MB")
    
    print(f"   📊 Total: {total_size:.2f} MB")
    return True

def check_models():
    """Verifica modelos treinados"""
    
    models_path = Path("models")
    
    if not models_path.exists():
        print("❌ Diretório models não encontrado")
        return False
    
    files = list(models_path.glob("*.pkl")) + list(models_path.glob("*.pt")) + list(models_path.glob("*.bin"))
    
    if not files:
        print("⚠️  Nenhum modelo encontrado em models")
        return False
    
    print(f"🤖 Models ({len(files)} arquivos):")
    
    total_size = 0
    for file in files:
        size_mb = file.stat().st_size / (1024 * 1024)
        total_size += size_mb
        print(f"   🎯 {file.name}: {size_mb:.2f} MB")
    
    print(f"   📊 Total: {total_size:.2f} MB")
    return True

def check_results():
    """Verifica resultados e métricas"""
    
    results_path = Path("results")
    
    if not results_path.exists():
        print("❌ Diretório results não encontrado")
        return False
    
    files = list(results_path.glob("*.json")) + list(results_path.glob("*.csv"))
    
    if not files:
        print("⚠️  Nenhum resultado encontrado em results")
        return False
    
    print(f"📊 Results ({len(files)} arquivos):")
    
    for file in files:
        size_kb = file.stat().st_size / 1024
        print(f"   📈 {file.name}: {size_kb:.1f} KB")
        
        # Se for JSON, tentar mostrar métricas
        if file.suffix == '.json':
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'accuracy' in data:
                        print(f"      ✅ Accuracy: {data['accuracy']:.3f}")
                    if 'f1_score' in data:
                        print(f"      📊 F1-Score: {data['f1_score']:.3f}")
            except:
                pass
    
    return True

def check_figures():
    """Verifica figuras geradas"""
    
    figures_path = Path("figures")
    
    if not figures_path.exists():
        print("❌ Diretório figures não encontrado")
        return False
    
    files = list(figures_path.glob("*.png")) + list(figures_path.glob("*.jpg")) + list(figures_path.glob("*.svg"))
    
    if not files:
        print("⚠️  Nenhuma figura encontrada em figures")
        return False
    
    print(f"🎨 Figures ({len(files)} arquivos):")
    
    total_size = 0
    for file in files:
        size_kb = file.stat().st_size / 1024
        total_size += size_kb
        print(f"   🖼️  {file.name}: {size_kb:.1f} KB")
    
    print(f"   📊 Total: {total_size/1024:.2f} MB")
    return True

def check_logs():
    """Verifica logs do sistema"""
    
    logs_path = Path("logs")
    
    if not logs_path.exists():
        print("❌ Diretório logs não encontrado")
        return False
    
    files = list(logs_path.glob("*.log"))
    
    if not files:
        print("⚠️  Nenhum log encontrado em logs")
        return False
    
    print(f"📋 Logs ({len(files)} arquivos):")
    
    for file in files:
        size_kb = file.stat().st_size / 1024
        print(f"   📝 {file.name}: {size_kb:.1f} KB")
        
        # Mostrar últimas linhas se for pequeno
        if size_kb < 100:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    if lines:
                        last_line = lines[-1].strip()
                        print(f"      🔍 Última linha: {last_line[:50]}...")
            except:
                pass
    
    return True

if __name__ == "__main__":
    print("📊 Status dos Dados - Pipeline Sentimento")
    print("=" * 45)
    
    # Verificar cada diretório
    checks = [
        ("Raw Data", check_data_raw),
        ("Processed Data", check_data_processed),
        ("Models", check_models),
        ("Results", check_results),
        ("Figures", check_figures),
        ("Logs", check_logs)
    ]
    
    results = {}
    
    for name, check_func in checks:
        print(f"\n🔍 {name}:")
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"❌ Erro ao verificar {name}: {e}")
            results[name] = False
    
    # Resumo final
    print("\n" + "=" * 45)
    print("📋 Resumo:")
    
    for name, status in results.items():
        emoji = "✅" if status else "❌"
        print(f"   {emoji} {name}")
    
    total_ok = sum(results.values())
    print(f"\n🎯 Status: {total_ok}/{len(results)} componentes OK")
    
    if total_ok == len(results):
        print("🎉 Pipeline completo!")
    else:
        print("⚠️  Pipeline incompleto - execute coleta e processamento")
