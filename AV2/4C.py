import numpy as np
import matplotlib.pyplot as plt
import control as ct

# Parâmetros base do sistema (Massa e Mola)
J = 0.25
K = 5.0

# 4b) Circuito Aberto COM atrito mecânico
B_mec = 0.05
num_aberto = [1]
den_aberto = [J, B_mec, K]
sys_aberto = ct.tf(num_aberto, den_aberto)

# 4c) Circuito Aberto SEM atrito (desprezível)
B_ideal = 0.0
num_ideal = [1]
den_ideal = [J, B_ideal, K]
sys_ideal = ct.tf(num_ideal, den_ideal)

# Vetor de tempo para simulação (0 a 10 segundos)
t = np.linspace(0, 10, 1000)

# Simulando a resposta ao degrau
t_ab, y_ab = ct.step_response(sys_aberto, T=t)
t_id, y_id = ct.step_response(sys_ideal, T=t)

# Configurações visuais do gráfico
plt.style.use('bmh')

# Criando a figura e corrigindo o título da janela
fig = plt.figure(figsize=(10, 5))
fig.canvas.manager.set_window_title('Simulação do Período Próprio')

# Plotagem das duas curvas para comparação
plt.plot(
    t_ab,
    y_ab,
    label=r'4b) Com atrito mecânico ($\zeta \approx 0.02$)',
    color='#1f77b4',
    linewidth=2
)

plt.plot(
    t_id,
    y_id,
    label=r'4c) Atrito desprezível ($\zeta = 0$)',
    color='#ff7f0e',
    linestyle='--',
    linewidth=2
)

# Título e identificação dos eixos
plt.title(
    'Determinação do Período Próprio (Simulação)',
    fontsize=14,
    fontweight='bold'
)

plt.xlabel('Tempo (s)', fontsize=12)
plt.ylabel('Deflexão Angular', fontsize=12)

# Legenda e grade
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()

# Exibindo o gráfico
plt.show()