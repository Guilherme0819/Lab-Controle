import numpy as np
import matplotlib.pyplot as plt
import control as ct

# Parâmetros base do sistema (Questão 2 original)
J = 0.25
B = 0.05
K = 5.0
E = 9.0
r = 4.0
Phi_m = 4.0

# Definindo os três casos de Resistência (R)
R_super = 1.0   # R < Rc (Superamortecido)
R_crit = 3.32   # R = Rc (Amortecimento Crítico)
R_sub = 48.0   # R > Rc (Subamortecido - ensaio original)

valores_R = [R_super, R_crit, R_sub]

labels = [
    'Superamortecido (R = 1.0 Ω)',
    'Amortecimento Crítico (Rc = 3.32 Ω)',
    'Subamortecido (R = 48.0 Ω)'
]

cores = ['#1f77b4', '#ff7f0e', '#2ca02c']

# Configuração do estilo do gráfico
plt.style.use('bmh')

# Criando a figura
fig = plt.figure(figsize=(10, 6))

# Corrigindo o título da janela
fig.canvas.manager.set_window_title('Influência de R no Sistema')

for R, label, cor in zip(valores_R, labels, cores):

    # Calculando os novos amortecimentos para cada R
    Req = r + R
    Bem = (Phi_m ** 2) / Req
    Btotal = B + Bem

    # Ganho de entrada da EDO
    Kin = (Phi_m / Req) * E

    # Criando a função de transferência do sistema
    # Equação: J*s^2 + Btotal*s + K = Kin
    num = [Kin]
    den = [J, Btotal, K]

    sys = ct.tf(num, den)

    # Simulando a resposta ao degrau
    t, y = ct.step_response(
        sys,
        T=np.linspace(0, 8, 1000)
    )

    # Normalizando a resposta para facilitar a comparação
    y_norm = y / y[-1]

    # Plotando a curva
    plt.plot(
        t,
        y_norm,
        label=label,
        color=cor,
        linewidth=2.5
    )

# Configurações visuais do gráfico
plt.axhline(
    1.0,
    color='black',
    linestyle='--',
    alpha=0.5,
    label='Regime Permanente (Normalizado)'
)

plt.title(
    'Influência de R no Comportamento Dinâmico',
    fontsize=14,
    fontweight='bold'
)

plt.xlabel('Tempo (s)', fontsize=12)

plt.ylabel(
    r'Deflexão Angular Normalizada $\theta(t)/\theta_{ss}$',
    fontsize=12
)

plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()

# Exibindo o gráfico
plt.show()