import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Definindo cada função de transferência com tuplas (numerador, denominador)
funcoes_transferencia = [
    ([62831], [1, 62831]),  # H(s) = 62831 / (s + 62831)
    ([3947840], [1, 889, 394784]),  # H(s) = 3947840 / (s^2 + 889s + 394784)
    ([1, 0], [1, 942]),  # H(s) = s / (s + 942)
    ([1, 0, 0, 0], [1, 37.7, 710.6, 6690]),  # H(s) = s^3 / (s^3 + 37.7s^2 + 710.6s + 6690)
    ([626126, 0], [1, 626126, 394784176]),  # H(s) = 626126s / (s^2 + 626126s + 394784176)
    ([1, 0, 395477191], [1, 625900, 395477191]),  # H(s) = (s^2 + 395477191) / (s^2 + 625900s + 395477191)
    (np.convolve([1, 0, 142129], [1, 0, 142129]), np.convolve([1, 38, 142129], [1, 38, 142129]))  # H(s) = (s^2 + 142129 / (s^2 + 38s + 142129))^3
]

# Faixa de frequência para os diagramas de Bode
w = np.logspace(0, 6, 1000)

# Função auxiliar para exibir H(s) em notação legível
def formatar_polinomio(coef):
    termos = [f"{a}*s^{len(coef)-i-1}" if i < len(coef)-1 else f"{a}" for i, a in enumerate(coef) if a != 0]
    return " + ".join(termos).replace("s^1", "s").replace("*s^0", "").replace(" 1*", " ")

for idx, (num, den) in enumerate(funcoes_transferencia):
    sistema = signal.TransferFunction(num, den)
    w, mag, phase = signal.bode(sistema, w)

    # Determina o tipo de filtro com base nos graus do numerador e denominador
    tipo_filtro = "Passa-baixa" if len(den) > len(num) else ("Passa-alta" if len(num) > len(den) else "Passa-banda")
    ganho_linear = num[0] / den[0] if tipo_filtro != "Passa-banda" else num[0] / den[0]
    ganho_db = 20 * np.log10(ganho_linear)

    # Calcula a frequência de corte para filtros passa-baixa e passa-alta
    if tipo_filtro == "Passa-baixa":
        freq_corte = den[-1]
    elif tipo_filtro == "Passa-alta":
        freq_corte = num[-1]
    else:
        freq_corte = [den[-1], num[-1]]  # Para passa-banda, duas frequências de corte
    
    # Exibir informações sobre o filtro
    print(f"\nFunção H(s) {idx + 1}: {formatar_polinomio(num)} / ({formatar_polinomio(den)})")
    print(f"Tipo de Filtro: {tipo_filtro}")
    print(f"Frequência(s) de Corte: {freq_corte} rad/s")
    print(f"Ganho em dB: {ganho_db:.2f} dB")

    # Plotar magnitude e fase do diagrama de Bode
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.semilogx(w, mag)
    plt.title(f'Diagrama de Bode - Função H(s) {idx + 1}')
    plt.xlabel('Frequência [rad/s]')
    plt.ylabel('Magnitude [dB]')

    plt.subplot(2, 1, 2)
    plt.semilogx(w, phase)
    plt.xlabel('Frequência [rad/s]')
    plt.ylabel('Fase [graus]')

    plt.tight_layout()
    plt.show()
