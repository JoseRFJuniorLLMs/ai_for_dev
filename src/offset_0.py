import numpy as np
import matplotlib.pyplot as plt

"""
Explicação dos Métodos de Quantização
FP32 (32-bit Floating Point): A representação original dos dados de ponto flutuante, com alta precisão e sem perda de dados.
FP16 (16-bit Floating Point): Reduz os dados para 16 bits, diminuindo o uso de memória e a precisão, mas é mais eficiente em termos de computação.
INT8 (8-bit Integer): Converte valores para inteiros de 8 bits, reduzindo drasticamente o tamanho do modelo. Este é um método comum em quantização pós-treinamento (PTQ).
Q8_4 (Quantization with Offset): Usa 8 bits para os valores e 4 bits para o offset, melhorando a precisão para distribuições de dados específicas.
Q4_0 (4-bit Quantization): Uma forma agressiva de quantização que reduz os valores para 4 bits, com maior perda de precisão.
Quantização Pós-Treinamento (PTQ): Aplicada após o treinamento do modelo, normalmente similar à quantização INT8, mas pode incluir ajustes para reduzir a perda de precisão.
Treinamento com Consciência de Quantização (QAT): O modelo é treinado levando em consideração a quantização, permitindo que o modelo se ajuste a possíveis perdas de precisão.
Quantização Dinâmica: Aplica a quantização apenas durante a inferência, sem alterar os pesos armazenados no modelo.
"""

# Valores de ponto flutuante de exemplo
float_values = np.array([0.15, -0.5, 0.9, 1.5, -1.2, 0.0, -0.75, 0.33])

# Funções de quantização
def quantize_no_offset(values, num_bits):
    scale = (np.max(values) - np.min(values)) / (2**num_bits - 1)
    quantized_values = np.round(values / scale).astype(int)
    return quantized_values, scale

def quantize_with_offset(values, num_bits, offset_bits):
    scale = (np.max(values) - np.min(values)) / (2**(num_bits - offset_bits) - 1)
    offset = np.min(values)
    quantized_values = np.round((values - offset) / scale).astype(int)
    return quantized_values, scale, offset

def dequantize(quantized_values, scale, offset=0):
    return quantized_values * scale + offset

# FP32 (32-bit Floating Point) - Representação Original
fp32_values = float_values.astype(np.float32)

# FP16 (16-bit Floating Point)
fp16_values = float_values.astype(np.float16)

# INT8 Quantization (8-bit Integer)
int8_quantized, scale_int8 = quantize_no_offset(float_values, 8)
int8_dequantized = dequantize(int8_quantized, scale_int8)

# Q8_4 Quantization (8-bit with 4-bit offset)
quant_with_offset, scale_with_offset, offset = quantize_with_offset(float_values, 8, 4)
dequant_with_offset = dequantize(quant_with_offset, scale_with_offset, offset)

# Q4_0 Quantization (4-bit quantization)
quant_q4_0, scale_q4_0 = quantize_no_offset(float_values, 4)
dequant_q4_0 = dequantize(quant_q4_0, scale_q4_0)

# Quantização Pós-Treinamento (PTQ)
ptq_quantized, scale_ptq = quantize_no_offset(float_values, 8)
ptq_dequantized = dequantize(ptq_quantized, scale_ptq)

# Treinamento com Consciência de Quantização (QAT)
qat_quantized, scale_qat = quantize_no_offset(float_values, 8)
qat_dequantized = dequantize(qat_quantized, scale_qat)

# Quantização Dinâmica
dynamic_quantized, scale_dynamic = quantize_no_offset(float_values, 8)
dynamic_dequantized = dequantize(dynamic_quantized, scale_dynamic)

# Plotando os valores originais e os dequantizados para comparação
methods = {
    'Original (FP32)': fp32_values,
    'FP16': fp16_values,
    'INT8': int8_dequantized,
    'Q8_4': dequant_with_offset,
    'Q4_0': dequant_q4_0,
    'PTQ': ptq_dequantized,
    'QAT': qat_dequantized,
    'Dynamic': dynamic_dequantized,
}

plt.figure(figsize=(12, 6))
x = np.arange(len(float_values))
width = 0.1

for i, (method, values) in enumerate(methods.items()):
    plt.bar(x + i * width, values, width, label=method)

plt.xticks(x + width * 4, [f'Val{i+1}' for i in range(len(float_values))])
plt.ylabel('Value')
plt.title('Comparison of Quantization Methods')
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()
