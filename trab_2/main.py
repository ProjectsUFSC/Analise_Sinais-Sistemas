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
    (np.convolve([1, 0, 142129], [1, 0, 142129]), np.convolve([1, 38, 142129], [1, 38, 142129]))  # H(s) = ((s^2 + 142129) / (s^2 + 38s + 142129))^3
]

# Lista dos tipos de filtros fornecidos
tipo_filtros = [
    "Filtro Passa-Baixas",    # sinal 1
    "Filtro Passa-Baixas",    # sinal 2
    "Filtro Passa-Altas",     # sinal 3
    "Filtro Passa-Altas",     # sinal 4
    "Filtro Passa-Faixa",     # sinal 5
    "Filtro Rejeita-Faixa",   # sinal 6
    "Filtro Rejeita-Faixa"    # sinal 7
]

# Faixa de frequência para os diagramas de Bode
w = np.logspace(0, 6, 1000)

# Função para encontrar frequências de corte (-3 dB)
def encontrar_freq_corte(w, mag, ganho_corte):
    indices = np.where(np.diff(np.sign(mag - ganho_corte)) != 0)[0]
    freq_corte = []
    for idx in indices:
        # Interpolação linear para estimar a frequência de corte
        w1, w2 = w[idx], w[idx+1]
        m1, m2 = mag[idx], mag[idx+1]
        if m2 != m1:  # Evitar divisão por zero
            wc = w1 + (ganho_corte - m1) * (w2 - w1) / (m2 - m1)
            freq_corte.append(wc)
    return freq_corte

for idx, (num, den) in enumerate(funcoes_transferencia):
    sistema = signal.TransferFunction(num, den)
    w, mag, phase = signal.bode(sistema, w)
    f = w / (2 * np.pi)  # Converter frequência para Hz

    # Obter o tipo de filtro a partir do cabarito
    tipo_filtro = tipo_filtros[idx]

    # Identificar frequências de corte (-3 dB abaixo do ganho máximo)
    ganho_max = max(mag)
    ganho_corte = ganho_max - 3  # -3 dB
    freq_cortes = encontrar_freq_corte(w, mag, ganho_corte)
    freq_cortes_hz = [fc / (2 * np.pi) for fc in freq_cortes]  # Converter para Hz

    # Plotar magnitude e fase do diagrama de Bode
    plt.figure(figsize=(10, 6))

    # Magnitude
    plt.subplot(2, 1, 1)
    plt.semilogx(f, mag, color='navy', label='Magnitude')
    # Plotar linhas de frequência de corte
    for fc in freq_cortes_hz:
        idx_fc = np.argmin(np.abs(f - fc))
        # Linha horizontal até a frequência de corte
        plt.semilogx(f[:idx_fc+1], [ganho_corte]*(idx_fc+1), color='orange', linestyle='--')
        # Linha vertical até o ponto de corte
        plt.semilogx([fc, fc], [min(mag), ganho_corte], color='orange', linestyle='--')
        # Marcar o ponto de corte
        plt.scatter([fc], [ganho_corte], color='orange', zorder=5)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.title(f"H(s) {idx + 1}")
    plt.xlabel('Frequência [Hz]')
    plt.ylabel('Magnitude [dB]')
    # Adicionar legendas
    plt.legend(['Magnitude', f'Frequência(s) de Corte: {[f"{fc:.2f}" for fc in freq_cortes_hz]} Hz', tipo_filtro, f"Ganho de Corte: {ganho_corte:.2f} dB"])

    # Fase
    plt.subplot(2, 1, 2)
    plt.semilogx(f, phase, color='green', label='Fase')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xlabel('Frequência [Hz]')
    plt.ylabel('Fase [graus]')
    plt.legend()

    plt.tight_layout()
    plt.show()