#!/usr/bin/env python3
"""
Universidade Federal do Amazonas - UFAM
Laboratório de Sistemas de Controle (LSC) - Avaliação 02
Questão 2: Simulação Analógica do Galvanômetro de Bobina Móvel

Este script implementa:
1. Simulação analógica estruturada (Somador, Integradores e Amplificadores/Ganhos)
2. Resposta temporal ao degrau de tensão E = 9V
3. Diagrama esquemático dos blocos analógicos
4. Mapa de polos e raízes características no plano s
5. Retrato no plano de fase (deflexão angular vs. velocidade angular)
6. Cálculo de todas as métricas: regime permanente, tempo de acomodação (2% e 5%),
   fator de amortecimento, frequência natural e sobressinal.
"""

import os
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ==============================================================================
# 1. PARÂMETROS DO SISTEMA (QUESTÃO 2)
# ==============================================================================
J = 0.25          # Momento de inércia [kg*m^2 ou N*s^2]
B = 0.05          # Coeficiente de atrito viscoso mecânico [N*m*s/rad ou N/m/s]
Phi_m = 4.0       # Fluxo magnético efetivo / acoplamento eletromagnético [Wb]
K = 5.0           # Constante elástica de torção da mola [N*m/rad ou N/m]
E = 9.0           # Tensão da fonte de alimentação [V]
rs = 1.5          # Resistência interna da bateria [ohms]
rg = 2.5          # Resistência interna da bobina do galvanômetro [ohms]
r = rs + rg       # Resistência interna total do instrumento [4.0 ohms]
R = 48.0          # Resistor sob medição [ohms]
Req = r + R       # Resistência equivalente total da malha [52.0 ohms]

# Coeficientes calculados
B_em = (Phi_m**2) / Req       # Amortecimento eletromagnético gerado pela fcem
B_total = B + B_em            # Amortecimento viscoso equivalente total
K_in = Phi_m / Req            # Ganho de entrada (V -> N*m)
Te_step = K_in * E            # Torque elétrico constante em regime

# Parâmetros canônicos de 2ª ordem
omega_n = np.sqrt(K / J)                     # Frequência natural não amortecida [rad/s]
zeta = B_total / (2.0 * J * omega_n)        # Fator de amortecimento
sigma = zeta * omega_n                      # Fator de atenuação (parte real) [1/s]
omega_d = omega_n * np.sqrt(1.0 - zeta**2)  # Frequência amortecida [rad/s]
Td = (2.0 * np.pi) / omega_d                # Período de oscilação [s]

# Polos do sistema
polo_1 = -sigma + 1j * omega_d
polo_2 = -sigma - 1j * omega_d

# Valor de regime permanente
theta_ss_rad = Te_step / K
theta_ss_deg = np.rad2deg(theta_ss_rad)

# Sobressinal analítico e tempos
Mp_percent = np.exp(-np.pi * zeta / np.sqrt(1.0 - zeta**2)) * 100.0
tp_theo = np.pi / omega_d
ts_2_theo = 4.0 / sigma
ts_5_theo = 3.0 / sigma

# ==============================================================================
# 2. MODELO DE SIMULAÇÃO ANALÓGICA (SOMADOR, INTEGRADOR E AMPLIFICADORES)
# ==============================================================================
def analog_simulation_deriv(t, y):
    """
    Representação estrita dos blocos de simulação analógica:
    - Estado y[0] = theta (saída do 2º integrador)
    - Estado y[1] = omega = d(theta)/dt (saída do 1º integrador)
    
    Blocos de Amplificação (Ganhos):
    - Amplificador de Entrada: K_in = Phi_m / (r + R)
    - Amplificador de Amortecimento: B_total = B + Phi_m^2 / (r + R)
    - Amplificador da Mola: K
    - Amplificador de Inércia: 1 / J
    
    Bloco Somador:
    - torque_net = + (K_in * E) - (B_total * omega) - (K * theta)
    
    Entrada do 1º Integrador:
    - d(omega)/dt = (1 / J) * torque_net
    """
    theta, omega = y
    
    # Amplificadores
    torque_eletrico = K_in * E            # Ganho de entrada aplicado ao degrau E
    torque_amortecimento = B_total * omega # Ganho de realimentação de velocidade
    torque_mola = K * theta               # Ganho de realimentação de posição
    
    # Somador
    torque_liquido = torque_eletrico - torque_amortecimento - torque_mola
    
    # Amplificador de inércia (1/J) alimentando o 1º Integrador
    d_omega_dt = (1.0 / J) * torque_liquido
    
    # Saída do 1º integrador alimenta o 2º Integrador
    d_theta_dt = omega
    
    return [d_theta_dt, d_omega_dt]

# Simulação numérica de alta fidelidade
t_final = 8.0
t_points = 20000
t_span = (0.0, t_final)
t_eval = np.linspace(0.0, t_final, t_points)
initial_state = [0.0, 0.0]

sol = integrate.solve_ivp(
    analog_simulation_deriv,
    t_span,
    initial_state,
    t_eval=t_eval,
    method='RK45',
    rtol=1e-10,
    atol=1e-12
)

t = sol.t
theta = sol.y[0]
omega = sol.y[1]
theta_deg = np.rad2deg(theta)

# ==============================================================================
# 3. EXTRAÇÃO DAS MÉTRICAS NUMÉRICAS DA RESPOSTA TEMPORAL
# ==============================================================================
# Valor final numérico
theta_ss_sim = theta[-1]

# Pico e Sobressinal numéricos
idx_peak = np.argmax(theta)
t_peak_sim = t[idx_peak]
theta_peak_sim = theta[idx_peak]
Mp_sim = ((theta_peak_sim - theta_ss_rad) / theta_ss_rad) * 100.0

# Tempo de Acomodação numérico (faixa de 2%)
tol_2 = 0.02 * theta_ss_rad
outside_2 = np.where(np.abs(theta - theta_ss_rad) > tol_2)[0]
ts_2_sim = t[outside_2[-1] + 1] if len(outside_2) > 0 else 0.0

# Tempo de Acomodação numérico (faixa de 5%)
tol_5 = 0.05 * theta_ss_rad
outside_5 = np.where(np.abs(theta - theta_ss_rad) > tol_5)[0]
ts_5_sim = t[outside_5[-1] + 1] if len(outside_5) > 0 else 0.0

# Envoltória exponencial analítica
envelope_sup = theta_ss_rad + (theta_ss_rad / np.sqrt(1.0 - zeta**2)) * np.exp(-sigma * t)
envelope_inf = theta_ss_rad - (theta_ss_rad / np.sqrt(1.0 - zeta**2)) * np.exp(-sigma * t)

# ==============================================================================
# 4. GERAÇÃO DOS GRÁFICOS
# ==============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.0

# ------------------------------------------------------------------------------
# FIGURA 1: RESPOSTA TEMPORAL AO DEGRAU DE TENSÃO
# ------------------------------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(11, 6), dpi=300)

color_rad = '#1f77b4'
color_deg = '#ff7f0e'

# Curva principal
line1 = ax1.plot(t, theta, color='#0b5394', linewidth=2.2, label=r'Deflexão $\theta(t)$ [rad] (Simulação)')

# Linha de regime permanente
ax1.axhline(theta_ss_rad, color='#274e13', linestyle='--', linewidth=1.8,
            label=f'Regime Permanente Teórico $\\theta_{{ss}} = {theta_ss_rad:.4f}$ rad')

# Faixa de tolerância de 2%
ax1.fill_between(t, theta_ss_rad * 0.98, theta_ss_rad * 1.02,
                 color='#93c47d', alpha=0.35, label='Faixa de Tolerância de 2% (Medição Precisa)')

# Faixa de tolerância de 5%
ax1.fill_between(t, theta_ss_rad * 0.95, theta_ss_rad * 1.05,
                 color='#ffd966', alpha=0.20, label='Faixa de Tolerância de 5%')

# Envoltória teórica
ax1.plot(t, envelope_sup, color='#cc0000', linestyle=':', linewidth=1.2, alpha=0.7, label='Envoltória Exponencial')
ax1.plot(t, envelope_inf, color='#cc0000', linestyle=':', linewidth=1.2, alpha=0.7)

# Marcação do Pico
ax1.plot(t_peak_sim, theta_peak_sim, 'ro', markersize=7)
ax1.annotate(f'Pico: {theta_peak_sim:.4f} rad ({np.rad2deg(theta_peak_sim):.2f}°)\n$M_p = {Mp_sim:.1f}\\%$ em $t_p = {t_peak_sim:.2f}$ s',
             xy=(t_peak_sim, theta_peak_sim), xytext=(t_peak_sim + 0.35, theta_peak_sim + 0.015),
             arrowprops=dict(facecolor='#cc0000', shrink=0.08, width=1.2, headwidth=6),
             fontsize=9.5, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='#fce5cd', edgecolor='#e69138'))

# Marcação do Tempo de Acomodação (2%)
ax1.plot(ts_2_sim, theta_ss_rad * 1.02, 's', color='#38761d', markersize=7)
ax1.annotate(f'Tempo de Acomodação (2%):\n$t_s = {ts_2_sim:.2f}$ s',
             xy=(ts_2_sim, theta_ss_rad * 1.02), xytext=(ts_2_sim - 1.6, theta_ss_rad + 0.04),
             arrowprops=dict(facecolor='#38761d', shrink=0.08, width=1.2, headwidth=6),
             fontsize=9.5, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='#d9ead3', edgecolor='#6aa84f'))

ax1.set_xlabel('Tempo $t$ (segundos)', fontsize=12, fontweight='bold')
ax1.set_ylabel(r'Deflexão Angular $\theta(t)$ (rad)', fontsize=12, fontweight='bold', color='#0b5394')
ax1.tick_params(axis='y', labelcolor='#0b5394')
ax1.set_xlim(0, t_final)
ax1.set_ylim(-0.02, 0.25)
ax1.grid(True, linestyle='--', alpha=0.6)

# Eixo secundário em graus
ax2 = ax1.twinx()
ax2.set_ylabel(r'Deflexão Angular $\theta(t)$ (graus)', fontsize=12, fontweight='bold', color='#a64d79')
ax2.tick_params(axis='y', labelcolor='#a64d79')
ax2.set_ylim(np.rad2deg(-0.02), np.rad2deg(0.25))

plt.title('Resposta ao Degrau do Galvanômetro de Bobina Móvel (Questão 2)\n' +
          f'$J=0.25$, $B=0.05$, $\\Phi_m=4.0$, $K=5.0$, $E=9$ V, $R=48\\ \\Omega$ $(\\zeta = {zeta:.4f}, \\omega_n = {omega_n:.2f}$ rad/s)',
          fontsize=13, fontweight='bold', pad=15)

ax1.legend(loc='lower right', framealpha=0.95, fontsize=8.5)
plt.tight_layout()
fig_resp_path = os.path.join(script_dir, 'grafico_q2_resposta_temporal.png')
plt.savefig(fig_resp_path)
plt.close()

# ------------------------------------------------------------------------------
# FIGURA 2: DIAGRAMA EM BLOCOS DA SIMULAÇÃO ANALÓGICA
# ------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 6.5), dpi=300)
ax.set_xlim(-1, 15)
ax.set_ylim(-4, 6)
ax.axis('off')

# Caixas / Blocos
def draw_block(ax, x, y, w, h, text, subtext="", bgcolor="#cfe2f3", edge="#0b5394"):
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                  facecolor=bgcolor, edgecolor=edge, linewidth=2)
    ax.add_patch(rect)
    if subtext:
        ax.text(x + w/2, y + h/2 + 0.22, text, ha='center', va='center', fontsize=10.5, fontweight='bold', color='#073763')
        ax.text(x + w/2, y + h/2 - 0.25, subtext, ha='center', va='center', fontsize=9, color='#274e13')
    else:
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=11, fontweight='bold', color='#073763')

def draw_circle(ax, x, y, r, text, bgcolor="#fff2cc", edge="#b45f06"):
    circle = plt.Circle((x, y), r, facecolor=bgcolor, edgecolor=edge, linewidth=2)
    ax.add_patch(circle)
    ax.text(x, y, text, ha='center', va='center', fontsize=14, fontweight='bold', color='#7f6000')

# Blocos
# 1. Entrada
ax.text(-0.5, 3.5, "Entrada Degrau\n$E = 9\\text{ V}$", ha='center', va='center', fontsize=10, fontweight='bold')
ax.annotate("", xy=(1.0, 3.5), xytext=(0.2, 3.5), arrowprops=dict(arrowstyle="->", lw=2, color="#0b5394"))

# 2. Amplificador de Entrada
draw_block(ax, 1.0, 2.7, 2.2, 1.6, "Amplificador\nGanho de Entrada", r"$K_e = \frac{\Phi_m}{r+R} = \frac{1}{13}$", "#d9ead3", "#38761d")
ax.annotate("", xy=(4.2, 3.5), xytext=(3.2, 3.5), arrowprops=dict(arrowstyle="->", lw=2, color="#0b5394"))
ax.text(3.7, 3.8, r"$T_e(t)$", ha='center', fontsize=10, fontweight='bold', color="#0b5394")

# 3. Somador
draw_circle(ax, 4.6, 3.5, 0.4, r"$\Sigma$", "#fff2cc", "#b45f06")
ax.text(4.25, 3.65, "+", fontsize=12, fontweight='bold', color="#38761d")
ax.text(4.45, 2.8, "−", fontsize=14, fontweight='bold', color="#cc0000")
ax.text(4.75, 2.8, "−", fontsize=14, fontweight='bold', color="#cc0000")

# 4. Saída do somador para Amplificador 1/J
ax.annotate("", xy=(5.8, 3.5), xytext=(5.0, 3.5), arrowprops=dict(arrowstyle="->", lw=2, color="#0b5394"))
ax.text(5.4, 3.85, r"$\tau_{\text{net}}$", ha='center', fontsize=10, fontweight='bold', color="#073763")

# 5. Amplificador 1/J
draw_block(ax, 5.8, 2.7, 1.8, 1.6, "Amplificador\nInércia", r"$\frac{1}{J} = 4.0$", "#fce5cd", "#b45f06")
ax.annotate("", xy=(8.4, 3.5), xytext=(7.6, 3.5), arrowprops=dict(arrowstyle="->", lw=2, color="#0b5394"))
ax.text(8.0, 3.85, r"$\ddot{\theta}(t)$", ha='center', fontsize=10, fontweight='bold', color="#073763")

# 6. Integrador 1 (produz velocidade omega)
draw_block(ax, 8.4, 2.7, 1.8, 1.6, "1º Integrador", r"$\int dt \quad \left(\frac{1}{s}\right)$", "#cfe2f3", "#0b5394")
ax.annotate("", xy=(11.0, 3.5), xytext=(10.2, 3.5), arrowprops=dict(arrowstyle="->", lw=2, color="#0b5394"))
ax.text(10.6, 3.85, r"$\dot{\theta}(t) = \omega$", ha='center', fontsize=10, fontweight='bold', color="#073763")

# Ponto de ramificação para velocidade
ax.plot(10.6, 3.5, 'o', color="#0b5394", markersize=6)

# 7. Integrador 2 (produz posição theta)
draw_block(ax, 11.0, 2.7, 1.8, 1.6, "2º Integrador", r"$\int dt \quad \left(\frac{1}{s}\right)$", "#cfe2f3", "#0b5394")
ax.annotate("", xy=(14.0, 3.5), xytext=(12.8, 3.5), arrowprops=dict(arrowstyle="->", lw=2, color="#0b5394"))
ax.text(13.4, 3.85, r"$\theta(t)$", ha='center', fontsize=11, fontweight='bold', color="#073763")
ax.text(14.3, 3.5, "Saída\n(Deflexão)", ha='left', va='center', fontsize=10.5, fontweight='bold')

# Ponto de ramificação para posição
ax.plot(13.4, 3.5, 'o', color="#0b5394", markersize=6)

# 8. Laço de Realimentação 1: Velocidade -> Amplificador de Amortecimento
# Linha desce de 10.6 para y = 0.5
ax.plot([10.6, 10.6], [3.5, 0.5], color="#cc0000", lw=2)
ax.plot([10.6, 8.5], [0.5, 0.5], color="#cc0000", lw=2)
draw_block(ax, 6.2, -0.3, 2.3, 1.6, "Amplificador\nAmortecimento", r"$B_{\text{total}} = 0.3577$", "#ead1dc", "#a64d79")
ax.annotate("", xy=(6.2, 0.5), xytext=(8.5, 0.5), arrowprops=dict(arrowstyle="<-", lw=2, color="#cc0000"))
ax.plot([6.2, 4.4], [0.5, 0.5], color="#cc0000", lw=2)
ax.annotate("", xy=(4.4, 3.1), xytext=(4.4, 0.5), arrowprops=dict(arrowstyle="->", lw=2, color="#cc0000"))
ax.text(4.1, 1.8, r"$\tau_B = B_{\text{total}}\dot{\theta}$", fontsize=9, fontweight='bold', color="#cc0000")

# 9. Laço de Realimentação 2: Posição -> Amplificador da Mola
# Linha desce de 13.4 para y = -2.5
ax.plot([13.4, 13.4], [3.5, -2.5], color="#8e7cc3", lw=2)
ax.plot([13.4, 8.5], [-2.5, -2.5], color="#8e7cc3", lw=2)
draw_block(ax, 6.2, -3.3, 2.3, 1.6, "Amplificador\nRigidez Mola", r"$K = 5.0$", "#d0e0e3", "#134f5c")
ax.annotate("", xy=(6.2, -2.5), xytext=(8.5, -2.5), arrowprops=dict(arrowstyle="<-", lw=2, color="#8e7cc3"))
ax.plot([6.2, 4.8], [-2.5, -2.5], color="#8e7cc3", lw=2)
ax.annotate("", xy=(4.8, 3.1), xytext=(4.8, -2.5), arrowprops=dict(arrowstyle="->", lw=2, color="#8e7cc3"))
ax.text(5.0, -1.2, r"$\tau_K = K\theta$", fontsize=9, fontweight='bold', color="#8e7cc3")

plt.title("Diagrama de Blocos de Simulação Analógica do Galvanômetro (Questão 2)\n" +
          r"Implementação por Somador ($\Sigma$), Integradores ($\int dt$) e Amplificadores de Ganho",
          fontsize=13, fontweight='bold', pad=10)

plt.tight_layout()
fig_diag_path = os.path.join(script_dir, 'diagrama_q2_simulacao_analogica.png')
plt.savefig(fig_diag_path)
plt.close()

# ------------------------------------------------------------------------------
# FIGURA 3: MAPA DE POLOS E ZEROS NO PLANO S
# ------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 7), dpi=300)

# Eixos Re e Im
ax.axhline(0, color='black', linewidth=1.2)
ax.axvline(0, color='black', linewidth=1.2)

# Polos
ax.plot([polo_1.real, polo_2.real], [polo_1.imag, polo_2.imag], 'rx', markersize=12, markeredgewidth=2.8,
        label=f'Polos Complexos Conjugados:\n$s_{{1,2}} = -{sigma:.4f} \\pm j{omega_d:.4f}$')

# Linhas de projeção
ax.plot([polo_1.real, 0], [polo_1.imag, polo_1.imag], 'r--', alpha=0.5)
ax.plot([polo_2.real, 0], [polo_2.imag, polo_2.imag], 'r--', alpha=0.5)
ax.plot([polo_1.real, polo_1.real], [polo_1.imag, 0], 'r--', alpha=0.5)

# Vetor da origem ao polo
ax.plot([0, polo_1.real], [0, polo_1.imag], color='#0b5394', linestyle='-', linewidth=1.8)
ax.text(polo_1.real/2 - 0.4, polo_1.imag/2, f'$\\omega_n = {omega_n:.2f}$ rad/s', color='#0b5394', fontweight='bold', fontsize=9.5)

# Ângulo theta_p = arccos(zeta)
theta_rad = np.arccos(zeta)
arc_angle = np.linspace(np.pi - theta_rad, np.pi, 50)
arc_r = 1.2
ax.plot(-arc_r * np.cos(arc_angle - np.pi), arc_r * np.sin(arc_angle - np.pi), color='#674ea7', lw=1.5)
ax.text(-1.5, 0.5, f'$\\theta_p = \\arccos(\\zeta) = {np.rad2deg(theta_rad):.1f}^\\circ$', color='#674ea7', fontsize=9, fontweight='bold')

# Semiplano estável sombreado
ax.axvspan(-6, 0, color='#d9ead3', alpha=0.25, label='Região de Estabilidade Assintótica (SPE)')

# Anotações
ax.annotate(f'$s_1 = -{sigma:.4f} + j{omega_d:.4f}$', xy=(polo_1.real, polo_1.imag),
            xytext=(polo_1.real - 2.8, polo_1.imag + 0.3),
            arrowprops=dict(facecolor='red', shrink=0.08, width=1, headwidth=5),
            fontsize=9.5, fontweight='bold', bbox=dict(boxstyle='round,pad=0.2', facecolor='#fff2cc'))

ax.annotate(f'$s_2 = -{sigma:.4f} - j{omega_d:.4f}$', xy=(polo_2.real, polo_2.imag),
            xytext=(polo_2.real - 2.8, polo_2.imag - 0.5),
            arrowprops=dict(facecolor='red', shrink=0.08, width=1, headwidth=5),
            fontsize=9.5, fontweight='bold', bbox=dict(boxstyle='round,pad=0.2', facecolor='#fff2cc'))

ax.set_xlim(-6, 2)
ax.set_ylim(-6, 6)
ax.set_xlabel('Eixo Real $\\sigma$ [s$^{-1}$]', fontsize=11, fontweight='bold')
ax.set_ylabel('Eixo Imaginário $j\\omega$ [rad/s]', fontsize=11, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.6)
ax.set_title('Mapa de Polos no Plano $s$ (Questão 2)\n' +
             f'Comportamento Subamortecido ($\\zeta = {zeta:.4f} < 1$)',
             fontsize=12, fontweight='bold', pad=12)
ax.legend(loc='lower left', framealpha=0.95, fontsize=9)

plt.tight_layout()
fig_polos_path = os.path.join(script_dir, 'grafico_q2_polos_zeros.png')
plt.savefig(fig_polos_path)
plt.close()

# ------------------------------------------------------------------------------
# FIGURA 4: RETRATO NO PLANO DE FASE (THETA vs OMEGA)
# ------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

ax.plot(theta_deg, omega, color='#45818e', linewidth=2.0, label='Trajetória de Estado $(\\theta, \\dot{\\theta})$')
ax.plot(theta_deg[0], omega[0], 'go', markersize=8, label=f'Início: $(0, 0)$')
ax.plot(theta_ss_deg, 0, 'r*', markersize=14, label=f'Ponto de Equilíbrio: $({theta_ss_deg:.2f}^\\circ, 0)$')

ax.set_xlabel(r'Deflexão Angular $\theta(t)$ (graus)', fontsize=11, fontweight='bold')
ax.set_ylabel(r'Velocidade Angular $\dot{\theta}(t)$ (rad/s)', fontsize=11, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.6)
ax.set_title('Plano de Fase: Espiral Estável Subamortecida (Questão 2)\n' +
             f'Trajetória convergente em torno de $\\theta_{{ss}} = {theta_ss_deg:.2f}^\\circ$',
             fontsize=12, fontweight='bold', pad=12)
ax.legend(loc='upper right', framealpha=0.95, fontsize=9.5)

plt.tight_layout()
fig_fase_path = os.path.join(script_dir, 'grafico_q2_plano_fase.png')
plt.savefig(fig_fase_path)
plt.close()

# ==============================================================================
# 5. IMPRESSÃO DO RELATÓRIO TÉCNICO NO TERMINAL
# ==============================================================================
print("=" * 78)
print("RELATÓRIO TÉCNICO DE SIMULAÇÃO - AVALIAÇÃO 02 - QUESTÃO 2")
print("=" * 78)
print(f"Parâmetros de Entrada:")
print(f"  J = {J:.2f} kg*m^2 | B = {B:.2f} N*m*s/rad | Phi_m = {Phi_m:.1f} Wb | K = {K:.1f} N*m/rad")
print(f"  E = {E:.1f} V | rs = {rs:.1f} ohms | rg = {rg:.1f} ohms -> r = {r:.1f} ohms | R = {R:.1f} ohms")
print(f"  Req = r + R = {Req:.1f} ohms")
print("-" * 78)
print(f"Parâmetros Intermediários da EDO:")
print(f"  B_em (amortecimento elétrico) = Phi_m^2 / Req = {B_em:.6f} N*m*s/rad")
print(f"  B_total (amortecimento total) = B + B_em     = {B_total:.6f} N*m*s/rad")
print(f"  Ganho de Entrada K_in        = Phi_m / Req   = {K_in:.6f} N*m/V")
print(f"  Torque do Degrau Te          = K_in * E      = {Te_step:.6f} N*m")
print("-" * 78)
print(f"Forma Canônica de 2ª Ordem:")
print(f"  Frequência natural não-amortecida (omega_n) = {omega_n:.6f} rad/s")
print(f"  Fator de amortecimento (zeta)               = {zeta:.6f}")
print(f"  Fator de atenuação (sigma = zeta*omega_n)   = {sigma:.6f} 1/s")
print(f"  Frequência natural amortecida (omega_d)     = {omega_d:.6f} rad/s")
print(f"  Período próprio de oscilação amortecida Td  = {Td:.6f} s")
print(f"  Polos do Sistema                            = {polo_1:.4f}, {polo_2:.4f}")
print("-" * 78)
print(f"Respostas dos Itens da Questão 2:")
print(f"a) Regime Permanente:")
print(f"   - Valor Teórico Esperado : {theta_ss_rad:.6f} rad  ({theta_ss_deg:.4f} graus)")
print(f"   - Valor Simulado         : {theta_ss_sim:.6f} rad  ({np.rad2deg(theta_ss_sim):.4f} graus)")
print(f"   - Erro Relativo          : {abs(theta_ss_sim - theta_ss_rad)/theta_ss_rad * 100:.6f}%")
print(f"b) Tipo de Comportamento:")
print(f"   - Classificação          : SUBAMORTECIDO (0 < zeta < 1, zeta = {zeta:.4f})")
print(f"   - Polos Características  : Complexos conjugados no SPE com Re(s) = -{sigma:.4f} < 0")
print(f"   - Sobressinal Mp         : {Mp_sim:.2f}% (Pico: {theta_peak_sim:.4f} rad / {np.rad2deg(theta_peak_sim):.2f}° em tp = {t_peak_sim:.2f} s)")
print(f"c) Tempo de Acomodação (Medição Precisa):")
print(f"   - Critério 2% (Simulado) : {ts_2_sim:.4f} segundos")
print(f"   - Critério 2% (Teórico)  : {ts_2_theo:.4f} segundos  (ts ~= 4 / sigma)")
print(f"   - Critério 5% (Simulado) : {ts_5_sim:.4f} segundos")
print(f"   - Critério 5% (Teórico)  : {ts_5_theo:.4f} segundos  (ts ~= 3 / sigma)")
print(f"d) Avaliação do Projeto do Medidor:")
print(f"   - Aceitável?             : NÃO")
print(f"   - Motivo Principal       : Sobressinal excessivo (60.1%) e tempo de acomodação muito lento (> 5.1 s)")
print(f"                              para um instrumento de bancada (onde o tolerável é <= 2.0 s)")
print("=" * 78)
print(f"Gráficos gerados com sucesso:")
print(f"  -> {fig_resp_path}")
print(f"  -> {fig_diag_path}")
print(f"  -> {fig_polos_path}")
print(f"  -> {fig_fase_path}")
print("=" * 78)
