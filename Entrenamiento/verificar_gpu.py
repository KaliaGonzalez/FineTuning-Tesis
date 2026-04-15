"""
Script para verificar que tu GPU NVIDIA está correctamente configurada
Ejecutar en PowerShell: python verificar_gpu.py
"""

import torch
import sys

print("=" * 60)
print("🔍 VERIFICACIÓN DE GPU NVIDIA")
print("=" * 60)

# 1. Verificar PyTorch
print(f"\n✅ PyTorch versión: {torch.__version__}")

# 2. Verificar CUDA
print(f"✅ CUDA disponible: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"✅ Versión de CUDA: {torch.version.cuda}")
    print(f"✅ cuDNN disponible: {torch.backends.cudnn.enabled}")
    print(f"✅ cuDNN versión: {torch.backends.cudnn.version()}")

    # 3. Listar GPUs
    print(f"\n🎮 Número de GPUs detectadas: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"   GPU {i}: {gpu_name} ({gpu_memory:.2f} GB)")

    # 4. GPU actual
    print(f"\n✅ GPU actual: {torch.cuda.current_device()}")
    print(f"✅ Nombre GPU actual: {torch.cuda.get_device_name()}")

    # 5. Memoria
    print(f"\n💾 Memoria GPU:")
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"   Usada: {allocated:.2f} GB")
    print(f"   Reservada: {reserved:.2f} GB")
    print(f"   Total: {total:.2f} GB")

    # 6. Test rápido
    print(f"\n🧪 Test de velocidad...")
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()

    import time

    start = time.time()
    for _ in range(100):
        z = torch.matmul(x, y)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    print(f"   100 multiplicaciones de matrices (1000x1000) en GPU:")
    print(f"   ⏱️ Tiempo: {elapsed:.4f} segundos")
    print(f"   ✅ GPU está funcionando correctamente!")

else:
    print("\n❌ GPU NO DETECTADA")
    print("\n⚠️ Posibles soluciones:")
    print("   1. Instala CUDA Toolkit: https://developer.nvidia.com/cuda-downloads")
    print("   2. Verifica que tu GPU es NVIDIA (nvidia-smi en cmd)")
    print("   3. Actualiza los drivers de GPU")
    print("   4. Reinstala PyTorch con CUDA:")
    print(
        "      pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    )

print("\n" + "=" * 60)
