# QINetMNLP: Quantum-Inspired Multilingual Neural Language Model

A quantum-inspired neural network for multilingual natural language processing, optimized for edge computing with support for African languages.

## Features

- **Quantum Interference Modulation (QIM1D)**: Novel position-dependent modulation for enhanced sequence modeling
- **Byte-level Processing**: UTF-8 encoding preservation for tonal and morphological features
- **Multi-task Learning**: Joint training on language modeling, language identification, and boundary detection
- **Edge-Optimized**: Only 43,085 parameters (~0.16 MB) suitable for resource-constrained devices
- **Multilingual Support**: Trained on Luo, Kikuyu, and Lubukusu languages

## Model Architecture

The QINetMNLP model consists of:
1. Embedding layer for byte tokens (0-255) plus CLS token
2. Quantum Interference Modulator (QIM1D) for position-dependent modulation
3. Four dilated causal convolution blocks with Squeeze-and-Excitation attention
4. Multi-task heads for language modeling, language ID, and boundary detection

## Performance

- **Language ID Accuracy**: 94.2% overall
- **Model Size**: 43,085 parameters (~0.16 MB)
- **Inference Speed**: ~10ms per sequence on CPU
- **Memory Usage**: <50MB during inference
- **Languages**: Luo (96.1% F1), Kikuyu (93.4% F1), Lubukusu (92.8% F1)

## Files

- `qimnlp.py`: Core model architecture with QIM1D module
- `multilingual_trainer.py`: Training script for multilingual setup
- `generate2.py`: Interactive text generation with language detection
- `adaptive_finetune.py`: Adaptive fine-tuning from PDF documents
- `QINetMNLP_Scientific_Paper.md`: Comprehensive technical documentation

## Usage

### Training
```bash
python multilingual_trainer.py
```

### Text Generation
```bash
python generate2.py
```

### Adaptive Fine-tuning
```bash
python adaptive_finetune.py
```

## Requirements

- PyTorch
- NumPy
- PyPDF2 (for PDF processing)

## Citation

If you use this work, please cite:
```
QINetMNLP: A Quantum-Inspired Multilingual Neural Language Model for Edge Computing
[Author Names], 2025
```

## License

MIT License
