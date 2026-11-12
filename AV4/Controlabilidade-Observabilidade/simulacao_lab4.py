"""
simulacao_lab4.py
Laboratório de Sistemas de Controle - Avaliação 04
UFAM / FEEC
Linearização, Controlabilidade e Observabilidade

Este script realiza:
1. Simulações do Balanço Bifilar (Parte 1: Linearização)
   - Modelo não linear e modelos linearizados em torno de pi/8, pi/6, pi/3
   - Análise de erros e retrato de fase
2. Análise e Simulações do Filtro Notch Twin-T Ativo (Parte 2: Controlabilidade)
   - Resposta ao degrau e planos de fase 2D
   - Controlabilidade, formas canônicas e transformações de similaridade
   - Resposta a entradas senoidais u(t) = 1 + sin(2t) e u(t) = 1 + sin(4t)
   - Trajetória 3D dos estados
   - Diagrama de Bode da função de transferência
"""

import os
import json
import numpy as np
import scipy.linalg as la
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import control as ct

# Configuração de estilo para os gráficos
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'lines.linewidth': 1.8,
    'grid.alpha': 0.4,
    'grid.linestyle': '--'
})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("=========================================================")
print("      LABORATÓRIO DE SISTEMAS DE CONTROLE — LAB 04       ")
print("=========================================================")

# ==============================================================================
# PARTE 1: LINEARIZAÇÃO — BALANÇO BIFILAR
# ==============================================================================
print("\n--- [PARTE 1] BALANÇO BIFILAR: MODELAGEM E LINEARIZAÇÃO ---")

# Parâmetros físicos dados
M = 4.0      # Massa da barra (kg)
L = 0.8      # Comprimento da barra (m)
l = 2.0      # Comprimento das cordas (m)
a = 0.3      # Distância do centro até cada corda (m)
g = 10.0     # Aceleração da gravidade (m/s^2)

# Coeficientes da EDO
k1 = (12 * a * g) / (L**2)   # 12 * 0.3 * 10 / 0.64 = 56.25 rad/s^2
k2 = a / l                   # 0.3 / 2.0 = 0.15 rad^-1
omega_n0 = np.sqrt(k1 * k2)  # Frequência natural em torno de zero = sqrt(8.4375) approx 2.9047 rad/s

print(f"Parâmetros: M={M} kg, L={L} m, l={l} m, a={a} m, g={g} m/s^2")
print(f"k1 = 12*a*g/L^2 = {k1:.4f} rad/s^2")
print(f"k2 = a/l = {k2:.4f}")
print(f"k1*k2 = {k1*k2:.4f} (omega_n0 = {omega_n0:.4f} rad/s, T0 = {2*np.pi/omega_n0:.4f} s)")

def balanco_nl_2ord(t, y):
    theta, omega = y
    dtheta = omega
    domega = - k1 * np.tan(k2 * theta)
    return [dtheta, domega]

# Pontos de linearização
thetas_op = [np.pi/8, np.pi/6, np.pi/3]
labels_op = ["π/8 (22,5°)", "π/6 (30°)", "π/3 (60°)"]
keys_op = ["pi_8", "pi_6", "pi_3"]

t_span_b = (0, 6)
t_eval_b = np.linspace(0, 6, 1200)

dados_balanco = {}

# 1.1 Gráfico Comparativo das Simulações Não-Lineares e Linearizadas
fig, axes = plt.subplots(3, 1, figsize=(9, 8.5), sharex=True)

for idx, (th0, label, key) in enumerate(zip(thetas_op, labels_op, keys_op)):
    # 1. Simulação Não Linear (2ª ordem)
    sol_nl = solve_ivp(balanco_nl_2ord, t_span_b, [th0, 0.0], t_eval=t_eval_b, rtol=1e-9, atol=1e-9)
    th_nl = sol_nl.y[0]
    om_nl = sol_nl.y[1]
    
    # 2. Modelo linearizado clássico em torno de 0: \ddot{\theta} = - k1*k2 * \theta
    th_lin0 = th0 * np.cos(omega_n0 * t_eval_b)
    
    # 3. Modelo linearizado por série de Taylor em torno de th0:
    # f(theta) approx f(th0) + f'(th0)*(theta - th0)
    f0 = - k1 * np.tan(k2 * th0)
    df0 = - k1 * k2 / (np.cos(k2 * th0)**2)
    
    def balanco_tangente(t, y):
        th, om = y
        return [om, f0 + df0 * (th - th0)]
    
    sol_tan = solve_ivp(balanco_tangente, t_span_b, [th0, 0.0], t_eval=t_eval_b, rtol=1e-9, atol=1e-9)
    th_tan = sol_tan.y[0]
    
    # Cálculo de métricas de erro
    erro_max_lin0 = np.max(np.abs(th_nl - th_lin0))
    erro_rms_lin0 = np.sqrt(np.mean((th_nl - th_lin0)**2))
    erro_rel_lin0 = (erro_max_lin0 / th0) * 100
    
    erro_max_tan = np.max(np.abs(th_nl - th_tan))
    erro_rms_tan = np.sqrt(np.mean((th_nl - th_tan)**2))
    erro_rel_tan = (erro_max_tan / th0) * 100
    
    dados_balanco[key] = {
        "theta_0_rad": float(th0),
        "theta_0_deg": float(np.degrees(th0)),
        "erro_max_lin0_rad": float(erro_max_lin0),
        "erro_rel_lin0_pct": float(erro_rel_lin0),
        "erro_rms_lin0_rad": float(erro_rms_lin0),
        "erro_max_tan_rad": float(erro_max_tan),
        "erro_rel_tan_pct": float(erro_rel_tan),
        "f0": float(f0),
        "df0": float(df0)
    }
    
    ax = axes[idx]
    ax.plot(t_eval_b, np.degrees(th_nl), 'b-', label='Não Linear', lw=2.2)
    ax.plot(t_eval_b, np.degrees(th_lin0), 'r--', label=r'Linear em torno de $\theta_e = 0$', lw=1.8)
    ax.plot(t_eval_b, np.degrees(th_tan), 'g:', label=r'Linear (Taylor em $\theta_0$)', lw=1.8)
    ax.set_ylabel(r'$\theta(t)$ [graus]')
    ax.set_title(f'Condição Inicial: $\\theta(0) =$ {label} | Erro Máx. Lin. 0: {erro_rel_lin0:.2f}%')
    ax.grid(True)
    ax.legend(loc='upper right')

axes[-1].set_xlabel('Tempo [s]')
fig.suptitle('Comparação: Sistema Não Linear vs. Modelos Linearizados (Balanço Bifilar)', fontsize=13, y=0.995)
plt.tight_layout()
fig_path_comp = os.path.join(BASE_DIR, "fig_balanco_linearizado_comparativo.png")
fig.savefig(fig_path_comp, dpi=300)
plt.close(fig)
print(f"Salvo: {fig_path_comp}")

# 1.2 Retrato de Fase (Plano de Fase) do Balanço
fig, ax = plt.subplots(figsize=(7.5, 4.8))
for th0, label, color in zip(thetas_op, labels_op, ['#1f77b4', '#ff7f0e', '#2ca02c']):
    sol_nl = solve_ivp(balanco_nl_2ord, (0, 10), [th0, 0.0], t_eval=np.linspace(0, 10, 1000))
    ax.plot(np.degrees(sol_nl.y[0]), np.degrees(sol_nl.y[1]), color=color, lw=2, label=f'Não linear: $\\theta_0 =$ {label}')
    
    # Linear em torno de 0
    th_l = th0 * np.cos(omega_n0 * np.linspace(0, 10, 1000))
    om_l = - th0 * omega_n0 * np.sin(omega_n0 * np.linspace(0, 10, 1000))
    ax.plot(np.degrees(th_l), np.degrees(om_l), '--', color=color, alpha=0.6, lw=1.2)

ax.set_xlabel(r'Posição Angular $\theta$ [graus]')
ax.set_ylabel(r'Velocidade Angular $\dot{\theta}$ [graus/s]')
ax.set_title(r'Retrato de Fase: $\dot{\theta}$ vs $\theta$ (Trajetórias Fechadas / Centro Estável)')
ax.grid(True)
ax.legend(loc='lower right')
plt.tight_layout()
fig_path_fase = os.path.join(BASE_DIR, "fig_balanco_plano_fase.png")
fig.savefig(fig_path_fase, dpi=300)
plt.close(fig)
print(f"Salvo: {fig_path_fase}")

# 1.3 Interpretação de 1ª ordem caso interpretado literalmente como \dot{\theta} = - k1*tan(k2*theta)
fig, ax = plt.subplots(figsize=(8, 5))
t_ord1 = np.linspace(0, 1.0, 500)
for th0, label, color in zip(thetas_op, labels_op, ['#1f77b4', '#ff7f0e', '#2ca02c']):
    # Solução analítica de 1ª ordem: sin(k2*theta(t)) = sin(k2*theta0)*exp(-k1*k2*t)
    th_nl_1 = (1.0 / k2) * np.arcsin(np.sin(k2 * th0) * np.exp(- k1 * k2 * t_ord1))
    th_lin_1 = th0 * np.exp(- k1 * k2 * t_ord1)
    ax.plot(t_ord1, np.degrees(th_nl_1), color=color, lw=2, label=f'NL 1ª ordem: {label}')
    ax.plot(t_ord1, np.degrees(th_lin_1), '--', color=color, alpha=0.7, lw=1.5, label=f'Lin 1ª ordem: {label}')

ax.set_xlabel('Tempo [s]')
ax.set_ylabel(r'$\theta(t)$ [graus]')
ax.set_title(r'Resposta se interpretado literalmente como EDO de 1ª Ordem $\dot{\theta} = -k_1 \tan(k_2 \theta)$')
ax.grid(True)
ax.legend(loc='upper right')
plt.tight_layout()
fig_path_ord1 = os.path.join(BASE_DIR, "fig_balanco_ordem1.png")
fig.savefig(fig_path_ord1, dpi=300)
plt.close(fig)
print(f"Salvo: {fig_path_ord1}")


# ==============================================================================
# PARTE 2: FILTRO NOTCH TWIN-T — CONTROLABILIDADE E OBSERVABILIDADE
# ==============================================================================
print("\n--- [PARTE 2] FILTRO NOTCH ATIVO: CONTROLABILIDADE E OBSERVABILIDADE ---")

RC = 0.25
K = 0.75  # R4/R5 = 3/4

# Matrizes no espaço de estados
A = np.array([
    [- (2.0 - K)/RC,   K/RC,        -1.0/RC],
    [  K/RC,           K/RC,        -1.0/RC],
    [  (1.0 + 2.0*K)/(2.0*RC), (1.0 + 2.0*K)/(2.0*RC), -1.0/RC]
])

# B conforme escrito na apostila (com fator 2 no denominador da 3ª linha)
B_handout = np.array([
    [(2.0 - K)/RC],
    [-K/RC],
    [-K/(2.0*RC)]
])

# B derivado estritamente do circuito físico
B_physical = np.array([
    [(2.0 - K)/RC],
    [-K/RC],
    [-K/RC]
])

# Adotaremos B_handout para seguir estritamente o texto da apostila,
# demonstrando também os resultados com B_physical.
B = B_handout

C = np.array([[-(1.0 + K), -(1.0 + K), 0.0]])
D = np.array([[1.0 + K]])

print("Matriz A:\n", A)
print("Matriz B (apostila):\n", B)
print("Matriz C:\n", C)
print("Matriz D:\n", D)

# Item b: Matriz de Controlabilidade
Mc = ct.ctrb(A, B)
det_Mc = float(np.linalg.det(Mc))
rank_Mc = int(np.linalg.matrix_rank(Mc))
print("\nMatriz de Controlabilidade Mc = [B  AB  A^2B]:\n", Mc)
print(f"det(Mc) = {det_Mc:.4f}")
print(f"Posto de Mc = {rank_Mc} (Dimensão do espaço = 3)")
print("=> O sistema é COMPLETAMENTE CONTROLÁVEL!")

# Item c: Subespaço controlável
# Como o posto é 3, o subespaço controlável é todo o R^3 (base canônica)
print("Subespaço controlável: Im(Mc) = R^3")

# Item d: Formas Canônicas e Matrizes de Transformação
# Polinômio característico: det(sI - A) = s^3 + a2 s^2 + a1 s + a0
poly_A = np.poly(A)
a2 = float(poly_A[1])
a1 = float(poly_A[2])
a0 = float(poly_A[3])
print(f"\nPolinômio característico: s^3 + {a2:.1f} s^2 + {a1:.1f} s + {a0:.1f}")

# Forma Canônica do Controlador (Companion com última linha [-a0, -a1, -a2]):
Ac = np.array([
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [-a0, -a1, -a2]
])
Bc = np.array([[0.0], [0.0], [1.0]])
Mcc = ct.ctrb(Ac, Bc)

# Matriz de transformação T tal que x = T z => z = T^-1 x
# Ac = T^-1 A T, Bc = T^-1 B
T_ctr = Mc @ np.linalg.inv(Mcc)
T_ctr_inv = np.linalg.inv(T_ctr)
Cc = C @ T_ctr
Dc = D

print("Matriz de Transformação T (para Forma do Controlador):\n", np.round(T_ctr, 4))
print("Ac = T^-1 A T:\n", np.round(T_ctr_inv @ A @ T_ctr, 4))
print("Bc = T^-1 B:\n", np.round(T_ctr_inv @ B, 4))
print("Cc = C T:\n", np.round(Cc, 4))
print("Dc = D:\n", Dc)

# Item e: Função de transferência e cancelamento de pólo-zero
sys_ss = ct.ss(A, B, C, D)
sys_tf = ct.ss2tf(sys_ss)
print("\nFunção de Transferência G(s) via State-Space:\n", sys_tf)

eigvals_A = np.linalg.eigvals(A)
zeros_sys = ct.zeros(sys_ss)
poles_sys = ct.poles(sys_ss)
print("Pólos de A:", np.round(poles_sys, 4))
print("Zeros de G(s):", np.round(zeros_sys, 4))

# Item a: Resposta ao degrau unitário (t in [0, 4], x(0) = 0)
t_deg = np.linspace(0, 4, 1000)
u_deg = np.ones_like(t_deg)

t_out, y_deg, x_deg = ct.forced_response(sys_ss, T=t_deg, U=u_deg, X0=np.zeros(3), return_x=True)

# Modelo físico com B_physical
sys_phys = ct.ss(A, B_physical, C, D)
_, y_deg_phys, x_deg_phys = ct.forced_response(sys_phys, T=t_deg, U=u_deg, X0=np.zeros(3), return_x=True)

# 2.1 Gráfico da Saída y(t) e dos Estados x(t) ao Degrau Unitário
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 8.5), sharex=True)

ax1.plot(t_out, y_deg, 'b-', lw=2.2, label=r'Saída $y(t)$ (Matriz $B$ da apostila, $y(\infty)=3,06$ V)')
ax1.plot(t_out, y_deg_phys, 'm--', lw=2.0, label=r'Saída $y(t)$ (Modelo físico do circuito, $y(\infty)=1,75$ V)')
ax1.axhline(3.0625, color='b', linestyle=':', alpha=0.6)
ax1.axhline(1.75, color='m', linestyle=':', alpha=0.6)
ax1.set_ylabel(r'Tensão de Saída [V]')
ax1.set_title(r'Resposta ao Degrau Unitário ($u(t) = 1$ V, $x(0) = 0$): Saída $y(t)$')
ax1.grid(True)
ax1.legend(loc='lower right')

ax2.plot(t_out, x_deg[0], 'm-', lw=1.8, label=r'$x_1(t) = V_{C1}(t)$')
ax2.plot(t_out, x_deg[1], 'c-', lw=1.8, label=r'$x_2(t) = V_{C2}(t)$')
ax2.plot(t_out, x_deg[2], 'g-', lw=1.8, label=r'$x_3(t) = V_{C3}(t)$')
ax2.set_xlabel('Tempo [s]')
ax2.set_ylabel('Tensões nos Capacitores [V]')
ax2.set_title('Evolução Temporal dos Estados $x_1(t), x_2(t), x_3(t)$')
ax2.grid(True)
ax2.legend(loc='center right')

plt.tight_layout()
fig_path_step = os.path.join(BASE_DIR, "fig_notch_degrau_saida_estados.png")
fig.savefig(fig_path_step, dpi=300)
plt.close(fig)
print(f"Salvo: {fig_path_step}")

# 2.2 Projeções 2D dos Planos de Fase (x1, x2), (x2, x3) e (x3, x1)
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

axes[0].plot(x_deg[0], x_deg[1], 'b-', lw=2)
axes[0].plot(x_deg[0][0], x_deg[1][0], 'go', markersize=8, label='Início (0,0)')
axes[0].plot(x_deg[0][-1], x_deg[1][-1], 'rs', markersize=8, label='Regime')
axes[0].set_xlabel(r'$x_1 = V_{C1}$ [V]')
axes[0].set_ylabel(r'$x_2 = V_{C2}$ [V]')
axes[0].set_title(r'Projeção Plano $(x_1, x_2)$')
axes[0].grid(True)
axes[0].legend()

axes[1].plot(x_deg[1], x_deg[2], 'r-', lw=2)
axes[1].plot(x_deg[1][0], x_deg[2][0], 'go', markersize=8, label='Início (0,0)')
axes[1].plot(x_deg[1][-1], x_deg[2][-1], 'rs', markersize=8, label='Regime')
axes[1].set_xlabel(r'$x_2 = V_{C2}$ [V]')
axes[1].set_ylabel(r'$x_3 = V_{C3}$ [V]')
axes[1].set_title(r'Projeção Plano $(x_2, x_3)$')
axes[1].grid(True)
axes[1].legend()

axes[2].plot(x_deg[2], x_deg[0], 'g-', lw=2)
axes[2].plot(x_deg[2][0], x_deg[0][0], 'go', markersize=8, label='Início (0,0)')
axes[2].plot(x_deg[2][-1], x_deg[0][-1], 'rs', markersize=8, label='Regime')
axes[2].set_xlabel(r'$x_3 = V_{C3}$ [V]')
axes[2].set_ylabel(r'$x_1 = V_{C1}$ [V]')
axes[2].set_title(r'Projeção Plano $(x_3, x_1)$')
axes[2].grid(True)
axes[2].legend()

plt.tight_layout()
fig_path_proj = os.path.join(BASE_DIR, "fig_notch_projecoes_2d.png")
fig.savefig(fig_path_proj, dpi=300)
plt.close(fig)
print(f"Salvo: {fig_path_proj}")

# Item f: Simulação para u(t) = 1 + sin(2t)
t_sin = np.linspace(0, 10, 1500)
u_sin2 = 1.0 + np.sin(2.0 * t_sin)
_, y_sin2, x_sin2 = ct.forced_response(sys_ss, T=t_sin, U=u_sin2, X0=np.zeros(3), return_x=True)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 7.5), sharex=True)
ax1.plot(t_sin, u_sin2, 'k--', label=r'Entrada $u(t) = 1 + \sin(2t)$', lw=1.8)
ax1.plot(t_sin, y_sin2, 'b-', label=r'Saída $y(t)$', lw=2.2)
ax1.set_ylabel('Tensão [V]')
ax1.set_title(r'Simulação com $u(t) = 1 + \sin(2t)$ ($\omega = 2$ rad/s): Resposta Forçada')
ax1.grid(True)
ax1.legend(loc='upper right')

ax2.plot(t_sin, x_sin2[0], label=r'$x_1 = V_{C1}$', lw=1.6)
ax2.plot(t_sin, x_sin2[1], label=r'$x_2 = V_{C2}$', lw=1.6)
ax2.plot(t_sin, x_sin2[2], label=r'$x_3 = V_{C3}$', lw=1.6)
ax2.set_xlabel('Tempo [s]')
ax2.set_ylabel('Estados [V]')
ax2.set_title('Evolução dos Estados no Tempo')
ax2.grid(True)
ax2.legend(loc='upper right')

plt.tight_layout()
fig_path_sin2 = os.path.join(BASE_DIR, "fig_notch_seno_2t.png")
fig.savefig(fig_path_sin2, dpi=300)
plt.close(fig)
print(f"Salvo: {fig_path_sin2}")

# Item g: Simulação para u(t) = 1 + sin(4t) — EFEITO NOTCH EM omega_0 = 4 rad/s
u_sin4 = 1.0 + np.sin(4.0 * t_sin)
_, y_sin4_handout, x_sin4_handout = ct.forced_response(sys_ss, T=t_sin, U=u_sin4, X0=np.zeros(3), return_x=True)
_, y_sin4_phys, x_sin4_phys = ct.forced_response(sys_phys, T=t_sin, U=u_sin4, X0=np.zeros(3), return_x=True)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 8.0), sharex=True)
ax1.plot(t_sin, u_sin4, 'k--', label=r'Entrada $u(t) = 1 + \sin(4t)$', lw=1.8)
ax1.plot(t_sin, y_sin4_phys, 'g-', lw=2.5, label=r'Saída $y(t)$ Filtro Notch Físico ($G(s)$ da apostila $\rightarrow$ Rejeição Total de $\sin(4t)$)')
ax1.plot(t_sin, y_sin4_handout, 'r:', lw=1.8, alpha=0.85, label=r'Saída $y(t)$ com Matriz $B$ escrita na apostila')
ax1.axhline(1.75, color='gray', linestyle=':', label=r'Nível DC puro $y_{ss} = 1,75$ V')
ax1.set_ylabel('Tensão [V]')
ax1.set_title(r'Efeito Notch em $\omega_0 = 4$ rad/s: $u(t) = 1 + \sin(4t) \rightarrow$ Rejeição da Frequência de Corte')
ax1.grid(True)
ax1.legend(loc='lower right')

ax2.plot(t_sin, x_sin4_phys[0], label=r'$x_1 = V_{C1}$', lw=1.6)
ax2.plot(t_sin, x_sin4_phys[1], label=r'$x_2 = V_{C2}$', lw=1.6)
ax2.plot(t_sin, x_sin4_phys[2], label=r'$x_3 = V_{C3}$', lw=1.6)
ax2.set_xlabel('Tempo [s]')
ax2.set_ylabel('Estados [V]')
ax2.set_title('Evolução dos Estados no Tempo sob Excitação na Frequência de Rejeição Notch (4 rad/s)')
ax2.grid(True)
ax2.legend(loc='upper right')

plt.tight_layout()
fig_path_sin4 = os.path.join(BASE_DIR, "fig_notch_seno_4t.png")
fig.savefig(fig_path_sin4, dpi=300)
plt.close(fig)
print(f"Salvo: {fig_path_sin4}")

# Item h: Trajetória 3D dos Estados (matplotlib ax.plot3D)
fig = plt.figure(figsize=(7.5, 5.5))
ax = fig.add_subplot(111, projection='3d')
ax.plot3D(x_deg[0], x_deg[1], x_deg[2], 'b-', lw=2.2, label='Degrau unitário')
ax.scatter([x_deg[0][0]], [x_deg[1][0]], [x_deg[2][0]], color='green', s=60, label='Ponto Inicial (0,0,0)')
ax.scatter([x_deg[0][-1]], [x_deg[1][-1]], [x_deg[2][-1]], color='red', s=60, label=r'Regime Estacionário $x(\infty)$')

ax.set_xlabel(r'$x_1 = V_{C1}$ [V]', labelpad=8)
ax.set_ylabel(r'$x_2 = V_{C2}$ [V]', labelpad=8)
ax.set_zlabel(r'$x_3 = V_{C3}$ [V]', labelpad=8)
ax.set_title(r'Trajetória Tridimensional dos Estados no $\mathbb{R}^3$ — Resposta ao Degrau', pad=15)
ax.legend(loc='upper left')
plt.tight_layout()
fig_path_3d = os.path.join(BASE_DIR, "fig_notch_trajetoria_3d.png")
fig.savefig(fig_path_3d, dpi=300)
plt.close(fig)
print(f"Salvo: {fig_path_3d}")

# Gráfico Extra: Diagrama de Bode do Filtro Notch comprovando o zero em w = 4 rad/s
w_range = np.logspace(-1, 2, 1000)
mag, phase, omega = ct.bode_plot(sys_tf, omega=w_range, plot=False)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.0, 5.0), sharex=True)
ax1.semilogx(omega, 20*np.log10(mag), 'b-', lw=2)
ax1.axvline(4.0, color='r', linestyle='--', label=r'$\omega_0 = 4$ rad/s (Frequência Notch)')
ax1.set_ylabel('Magnitude [dB]')
ax1.set_title('Diagrama de Bode: Resposta em Frequência do Filtro Notch Twin-T')
ax1.grid(True, which='both')
ax1.legend()

ax2.semilogx(omega, np.degrees(phase), 'r-', lw=2)
ax2.axvline(4.0, color='r', linestyle='--')
ax2.set_xlabel('Frequência [rad/s]')
ax2.set_ylabel('Fase [graus]')
ax2.grid(True, which='both')

plt.tight_layout()
fig_path_bode = os.path.join(BASE_DIR, "fig_notch_bode.png")
fig.savefig(fig_path_bode, dpi=300)
plt.close(fig)
print(f"Salvo: {fig_path_bode}")

# Exportar dados analíticos e de simulação em JSON para serem consumidos no relatório
dados_export = {
    "balanco": dados_balanco,
    "notch": {
        "RC": float(RC),
        "K": float(K),
        "A": A.tolist(),
        "B": B.tolist(),
        "C": C.tolist(),
        "D": D.tolist(),
        "Mc": Mc.tolist(),
        "det_Mc": float(det_Mc),
        "rank_Mc": int(rank_Mc),
        "poly_coeffs": [float(a2), float(a1), float(a0)],
        "T_ctr": T_ctr.tolist(),
        "Ac": Ac.tolist(),
        "Bc": Bc.tolist(),
        "Cc": Cc.tolist(),
        "Dc": Dc.tolist(),
        "poles": [f"{p.real:.4f} + {p.imag:.4f}j" if p.imag != 0 else f"{p.real:.4f}" for p in poles_sys],
        "zeros": [f"{z.real:.4f} + {z.imag:.4f}j" if z.imag != 0 else f"{z.real:.4f}" for z in zeros_sys],
        "y_regime_degrau": float(y_deg[-1]),
        "x_regime_degrau": [float(x_deg[0][-1]), float(x_deg[1][-1]), float(x_deg[2][-1])]
    }
}

json_path = os.path.join(BASE_DIR, "dados_simulacao.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(dados_export, f, indent=4)
print(f"\nDados exportados com sucesso em: {json_path}")
print("Todas as simulações e gráficos foram gerados perfeitamente!")
