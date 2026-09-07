import numpy as np
import matplotlib.pyplot as plt
import control as ct

# Valores calculados do sistema
A = [[-10, 1], [-0.02, -2]]
B = [[0], [2]]
C = [[1, 0]]
D = [[0]]

# Criação do modelo em Espaço de Estados
sys = ct.ss(A, B, C, D)

# Simulação das respostas (Degrau e Impulso)
t_step, y_step = ct.step_response(sys)
t_imp, y_imp = ct.impulse_response(sys)

# Configuração visual do Dashboard de resultados
plt.style.use('bmh')
fig = plt.figure(figsize=(12, 8))
fig.canvas.manager.set_window_title('Simulação do Sistema - Motor CC')
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.8])

# --- ÁREA SUPERIOR: TEXTOS E EQUAÇÕES ---
ax_text = fig.add_subplot(gs[0, :])
ax_text.axis('off')

titulo = "Modelagem e Simulação do Motor CC"
eq_gs = r"Função de Transferência: $G(s) = \frac{0.01}{0.005s^2 + 0.06s + 0.1001}$"
eq_ss = r"Espaço de Estados: $\dot{x} = Ax + Bu \quad | \quad y = Cx$"

matriz_A = "A = [[-10.00,  1.00]\n     [-0.02, -2.00]]"
matriz_B = "B = [[0.00]\n     [2.00]]"
matriz_C = "C = [[1.00, 0.00]]"

ax_text.text(0.5, 0.95, titulo, fontsize=18, fontweight='bold', ha='center', va='center')
ax_text.text(0.5, 0.65, eq_gs, fontsize=15, ha='center', va='center', color='#333333')
ax_text.text(0.5, 0.40, eq_ss, fontsize=15, ha='center', va='center', color='#333333')
ax_text.text(0.35, 0.10, matriz_A, fontsize=12, ha='center', va='center', family='monospace')
ax_text.text(0.55, 0.10, matriz_B, fontsize=12, ha='center', va='center', family='monospace')
ax_text.text(0.70, 0.10, matriz_C, fontsize=12, ha='center', va='center', family='monospace')

# --- GRÁFICO 1: RESPOSTA AO DEGRAU ---
ax1 = fig.add_subplot(gs[1, 0])
ax1.plot(t_step, y_step, color='#1f77b4', linewidth=2.5)
ax1.set_title('Resposta ao Degrau Unitário', fontsize=13, fontweight='bold')
ax1.set_xlabel('Tempo (s)', fontsize=11)
ax1.set_ylabel(r'Velocidade Angular $\omega(t)$ [rad/s]', fontsize=11)
ax1.fill_between(t_step, y_step, alpha=0.2, color='#1f77b4')

# --- GRÁFICO 2: RESPOSTA AO IMPULSO ---
ax2 = fig.add_subplot(gs[1, 1])
ax2.plot(t_imp, y_imp, color='#d62728', linewidth=2.5)
ax2.set_title('Resposta ao Impulso Unitário', fontsize=13, fontweight='bold')
ax2.set_xlabel('Tempo (s)', fontsize=11)
ax2.set_ylabel(r'Velocidade Angular $\omega(t)$ [rad/s]', fontsize=11)
ax2.fill_between(t_imp, y_imp, alpha=0.2, color='#d62728')

# Exibe a janela com os gráficos
plt.tight_layout()
plt.show()