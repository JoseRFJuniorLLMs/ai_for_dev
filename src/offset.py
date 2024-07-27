import numpy as np
import matplotlib.pyplot as plt

"""
Explicação dos Métodos de Quantização e seus Efeitos:

FP32 (32-bit Floating Point):
- Alta precisão, sem perda de dados
- Alto consumo de memória e processamento
- Benchmark para comparação

FP16 (16-bit Floating Point):
- Precisão reduzida
- Menor uso de memória (50% de FP32)
- Computação mais eficiente em hardware compatível

INT8 (8-bit Integer):
- Precisão significativamente reduzida
- Uso de memória muito baixo (25% de FP32)
- Computação muito eficiente, comum em inferência em dispositivos móveis

Q4_0 (4-bit Quantization):
- Precisão drasticamente reduzida
- Uso mínimo de memória (12.5% de FP32)
- Computação extremamente eficiente, mas com potencial perda significativa de qualidade

Quantização Dinâmica:
- Precisão variável dependendo dos dados de entrada
- Uso de memória reduzido durante a inferência
- Compromisso entre eficiência e precisão
"""

# Função para simular o comportamento do modelo
def model_behavior(x, quantization_method):
    if quantization_method == 'FP32':
        return np.sin(x) + 0.1 * np.random.randn(len(x))
    elif quantization_method == 'FP16':
        return np.sin(x).astype(np.float16) + 0.1 * np.random.randn(len(x)).astype(np.float16)
    elif quantization_method == 'INT8':
        return np.round(np.sin(x) * 127).astype(np.int8) / 127
    elif quantization_method == 'Q4_0':
        return np.round(np.sin(x) * 7).astype(np.int8) / 7
    elif quantization_method == 'Dynamic':
        # Simulando quantização dinâmica com precisão variável
        dynamic_precision = np.random.choice([8, 16, 32], size=len(x))
        return np.where(dynamic_precision == 32, np.sin(x),
                        np.where(dynamic_precision == 16, np.sin(x).astype(np.float16),
                                 np.round(np.sin(x) * 127).astype(np.int8) / 127))

# Gerando dados
x = np.linspace(0, 4*np.pi, 1000)

# Simulando comportamento do modelo com diferentes métodos de quantização
methods = ['FP32', 'FP16', 'INT8', 'Q4_0', 'Dynamic']
results = {method: model_behavior(x, method) for method in methods}

# Plotando resultados
plt.figure(figsize=(15, 10))
for i, (method, y) in enumerate(results.items()):
    plt.subplot(3, 2, i+1)
    plt.plot(x, y, label=f'{method} Output')
    plt.plot(x, np.sin(x), 'r--', label='Ideal Output', alpha=0.5)
    plt.title(f'{method} Quantization')
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.legend()
    plt.grid(True)

# Adicionando informações sobre consumo de recursos e precisão
resource_info = {
    'FP32': {'Memory': '100%', 'Computation': '100%', 'Precision': 'High'},
    'FP16': {'Memory': '50%', 'Computation': '60%', 'Precision': 'Medium'},
    'INT8': {'Memory': '25%', 'Computation': '30%', 'Precision': 'Low'},
    'Q4_0': {'Memory': '12.5%', 'Computation': '15%', 'Precision': 'Very Low'},
    'Dynamic': {'Memory': '25-100%', 'Computation': '30-100%', 'Precision': 'Variable'}
}

plt.subplot(3, 2, 6)
for i, method in enumerate(methods):
    plt.text(0, 1 - i*0.2, f"{method}:", fontweight='bold')
    plt.text(0.3, 1 - i*0.2, f"Memory: {resource_info[method]['Memory']}")
    plt.text(0.6, 1 - i*0.2, f"Computation: {resource_info[method]['Computation']}")
    plt.text(0.9, 1 - i*0.2, f"Precision: {resource_info[method]['Precision']}")
plt.axis('off')
plt.title('Resource Usage and Precision Comparison')

plt.tight_layout()
plt.show()

# Cálculo do erro médio quadrático (MSE) para cada método
mse = {method: np.mean((y - np.sin(x))**2) for method, y in results.items()}

print("Mean Squared Error (MSE) for each quantization method:")
for method, error in mse.items():
    print(f"{method}: {error:.6f}")