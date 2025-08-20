Se utilizó un modelo basado en el algoritmo U-Net 2D, donde se analiza a través de una red convolucional el espectrograma en 2D de un dataset (LibriSpeech) al que se le agrega ruido blanco de manera artificial, con el objetivo de entrenar al modelo para eliminar el ruido de las señales de voz y generar una versión limpia. La evaluación del modelo se realizó comparando la señal procesada con la señal limpia de referencia usando métricas objetivas como SNR y SDR, obteniendo mejoras significativas en la relación señal/ruido.
# Supresor de Ruido de Voz con U-Net 2D


## Características

- Arquitectura U-Net 2D 
- Procesamiento basado en espectrogramas STFT
- Pipeline completo de ML: datos, entrenamiento, evaluación
- Métricas estándar: SNR y SDR
- Soporte para ruido Blanco

## Instalación

Asegurese de tener instaladas la librerias : torch, torchaudio, numpy, matplotlib, tqdm
y sugiero utilizar un env nuevo hecho con conda.



## Uso
el archivo infer.py permite cargar un audio con ruido blanco para ser procesado por el modelo

```bash
infer.py --input "ej2.R.wav"
```

### Entrenamiento
el archivo train.py permite el entrenamiento escogiendo los diferentes parametros necesarios
```bash
train.py --root_dir LibriSpeech/dev-clean \
  --epochs 10 --batch_size 2 \
  --fixed_length_sec 2.0 --n_fft 256 --hop_length 128 --win_length 256 \
  --base_channels 16 --num_workers 0
```

### Evaluación
```bash
python scripts/evaluate.py --test_dir data/test/ --model_path models/best_model.pth
```

## Arquitectura Técnica

### Modelo
- **Entrada**:  dataset de pares de audio (clean, noisey)
- **Arquitectura**: U-Net 2D convolucional
- **Salida**: pesos en formato .pt que al ser ejecutados desde infer.py generan audio y graficas del proceso
- **Función de pérdida**: adam

### Dataset
- **Fuentes**: LibriSpeech
- **Ruido sintético**: Blanco
- **Preprocesamiento**: STFT, normalización logarítmica

### Métricas
- **SNR**: SIGNAL TO NOISE RATIO (objetivo: >10)
- **SDR**: Relación señal-distorsión (objetivo: >10 dB)
