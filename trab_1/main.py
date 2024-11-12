import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

# Parâmetros gerais
T = 2  # Período do sinal em segundos
dt = 0.001  # Intervalo de tempo
f_max = 25  # Frequência máxima para o espectro
n_values = [3, 5, 10, 25, 50, 100]  # Números de termos da série

# Tempo discretizado
t = np.arange(0, T, dt)

# Definindo os sinais
def sinal_a(t):
    return np.where((t >= 0) & (t < 1), 1, 0)

def sinal_b(t):
    return np.where((t >= 0) & (t < 1), 4 * t - 2, -4 * t + 6)

def sinal_c(t):
    return (2/3) * t - 1

def sinal_d(t):
    return np.where((t >= 0) & (t < 1), 1 + np.sin(2 * np.pi * t), 0)

# Lista dos sinais e nomes
sinais = [sinal_a, sinal_b, sinal_c, sinal_d]
nomes_sinais = ['Sinal A', 'Sinal B', 'Sinal C', 'Sinal D']

# Função para calcular coeficientes de Fourier
def calcular_coeficientes_fourier(x_t, T, N_termos):
    a0 = (2 / T) * np.sum(x_t * dt)  # Cálculo de a0
    an = np.zeros(N_termos)
    bn = np.zeros(N_termos)

    for n in range(1, N_termos + 1):
        cos_comp = np.cos(2 * np.pi * n * t / T)
        sin_comp = np.sin(2 * np.pi * n * t / T)
        an[n-1] = (2 / T) * np.sum(x_t * cos_comp * dt)
        bn[n-1] = (2 / T) * np.sum(x_t * sin_comp * dt)

    return a0, an, bn

# Calcular os coeficientes Cn e θn
def calcular_coeficientes_compactos(an, bn):
    Cn = np.sqrt(an**2 + bn**2)
    theta_n = np.arctan2(bn, an)
    return Cn, theta_n

# Função principal
for sinal_func, nome_sinal in zip(sinais, nomes_sinais):
    x_t = sinal_func(t)

    # Gráfico do sinal original, reconstruído e espectro
    for N_termos in n_values:
        # Calcular os coeficientes de Fourier
        a0, an, bn = calcular_coeficientes_fourier(x_t, T, N_termos)

        # Calcular e plotar o espectro da série compacta
        Cn, theta_n = calcular_coeficientes_compactos(an, bn)

        if N_termos == 100:
            # Criar tabela dos coeficientes
            tabela = []
            tabela.append(["Termo (n)", "a_n", "b_n", "C_n", "θ_n (rad)"])
            tabela.append([0, a0, 0, a0 / 2, 0])
            for n in range(1, N_termos + 1):
                tabela.append([n, an[n-1], bn[n-1], Cn[n-1], theta_n[n-1]])

            # Imprimir a tabela usando tabulate
            print(f"Tabela de coeficientes para {nome_sinal} com {N_termos} termos:\n")
            print(tabulate(tabela, headers="firstrow", tablefmt="grid"))
            print("\n" + "-"*60 + "\n")

        # Reconstruir o sinal usando os coeficientes
        x_reconstruido = np.ones_like(t) * a0 / 2
        for n in range(1, N_termos + 1):
            x_reconstruido += an[n-1] * np.cos(2 * np.pi * n * t / T) + bn[n-1] * np.sin(2 * np.pi * n * t / T)

        # Plotar o sinal original e reconstruído, junto com o espectro
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))  # Gráficos lado a lado
        
        # Sinal original e reconstruído
        axs[0].plot(t, x_t, label='Sinal Original', linestyle='--', color='blue', linewidth=2)
        axs[0].plot(t, x_reconstruido, label=f'{N_termos} Termos')
        axs[0].set_xlabel('Tempo (s)', fontsize=14)
        axs[0].set_ylabel('Amplitude', fontsize=14)
        axs[0].set_title(f'Reconstrução de {nome_sinal} com {N_termos} Termos')
        axs[0].legend(loc='upper right', fontsize=12)

        # Espectro da série compacta
        freqs = np.arange(1, len(Cn) + 1) / T
        axs[1].stem(freqs, Cn, basefmt=" ", linefmt='purple')
        axs[1].set_xlim(0, f_max)
        axs[1].set_xlabel('Frequência (Hz)')
        axs[1].set_ylabel('|C_n|')
        axs[1].set_title(f'Espectro da Série de Fourier Compacta para {nome_sinal}')

        plt.tight_layout()
        plt.show()
