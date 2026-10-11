#!/usr/bin/env python3
"""
Laboratório de Sistemas de Controle - UFAM
Experimento 1: Comportamento dinâmico de sistemas de 1ª e 2ª ordem
Questão 2 - Letra (d): Resposta ao Pulso como Aproximação do Impulso Unitário

Este script realiza a simulação do circuito com os parâmetros da Simulação 6:
R1 = 10 Ohm, R2 = 50 Ohm, C = 0.25 F, mu = 2.0, R3 = 200 Ohm
Para entradas do tipo pulso retangular com amplitude A = 1/L e largura L nos valores:
L = 32, 8, 2, 0.25, 0.0625 s.

Ele compara as respostas dinâmicas com a resposta ao impulso unitário ideal h(t),
demonstra a aproximação analítica via Série de Taylor, gera tabelas de convergência
e salva gráficos em alta resolução.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import control as ct

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Parâmetros fixos da Simulação 6
R1 = 10.0    # Ohms
R2 = 50.0    # Ohms
C  = 0.25    # Farads
mu = 2.0
R3 = 200.0   # Ohms

# Cálculo dos coeficientes do modelo da Simulação 6
denom = (R1 * R2 + R2 * R3 + R3 * R1) * C
a = ((1.0 - mu) * R1 + R3) / denom
b = (mu * R3) / denom
tau = 1.0 / a
K = b / a

# Resposta teórica ao impulso unitário: h(t) = b * exp(-a * t) = (K/tau) * exp(-t/tau)
h0 = b  # Valor inicial do impulso ideal h(0+)

# Valores de L para cada simulação da tabela
simulacoes_pulso = [
    {"sim": 1, "L": 32.0},
    {"sim": 2, "L": 8.0},
    {"sim": 3, "L": 2.0},
    {"sim": 4, "L": 0.25},
    {"sim": 5, "L": 0.0625},
]

# Modelo LTI no control: G(s) = b / (s + a)
sys = ct.tf([b], [1.0, a])

# Vetor de tempo global para análise macroscópica
t_max = 80.0
pontos_base = 4000

# ==============================================================================
# 1. PROCESSAMENTO E CÁLCULOS MATEMÁTICOS (EXATO, SIMULADO E TAYLOR)
# ==============================================================================
dados_processados = []

for item in simulacoes_pulso:
    L = item["L"]
    A = 1.0 / L
    razao_L_tau = L / tau

    # Valor de pico analítico exato em t = L:
    # y(L) = (K / L) * (1 - exp(-L / tau))
    y_pico_exato = (K / L) * (1.0 - np.exp(-L / tau))

    # Aproximação de Taylor de 1ª ordem: y(L) approx b * (1 - L / (2 * tau))
    y_taylor_1 = b * (1.0 - L / (2.0 * tau))

    # Aproximação de Taylor de 2ª ordem: y(L) approx b * (1 - L / (2*tau) + L^2 / (6*tau^2))
    y_taylor_2 = b * (1.0 - L / (2.0 * tau) + (L**2) / (6.0 * (tau**2)))

    # Erro relativo percentual do pico frente ao impulso ideal h(0+) = b
    erro_rel_h0 = abs(y_pico_exato - h0) / h0 * 100.0

    # Classificação da aproximação
    if razao_L_tau > 1.0:
        classif = "Pulso Longo (Comportamento de Degrau)"
    elif razao_L_tau > 0.2:
        classif = "Transição (Pulso Moderado)"
    elif razao_L_tau > 0.05:
        classif = "Boa Aproximação do Impulso"
    else:
        classif = "Excelente / Indistinguível do Impulso"

    # Vetor de tempo com espaçamento perfeitamente uniforme (dt pequeno para capturar L = 0.0625 s com alta precisão)
    dt = min(0.0005, L / 100.0)
    num_pontos = int(np.ceil(t_max / dt)) + 1
    t_sim = np.linspace(0, t_max, num_pontos)

    # Sinal de entrada u(t): pulso retangular
    u_sim = np.where((t_sim >= 0) & (t_sim <= L), A, 0.0)

    # Simulação usando control.forced_response com T uniformemente espaçado
    sim_resp = ct.forced_response(sys, T=t_sim, U=u_sim)
    t_out = sim_resp.time
    y_sim = sim_resp.outputs

    # Obter valor de pico numérico simulado
    idx_L = np.argmin(np.abs(t_out - L))
    y_pico_sim = y_sim[idx_L]

    dados_processados.append({
        "sim": item["sim"],
        "L": L,
        "A": A,
        "razao_L_tau": razao_L_tau,
        "y_pico_exato": y_pico_exato,
        "y_pico_sim": y_pico_sim,
        "y_taylor_1": y_taylor_1,
        "y_taylor_2": y_taylor_2,
        "erro_rel_h0": erro_rel_h0,
        "classif": classif,
        "t_sim": t_sim,
        "u_sim": u_sim,
        "y_sim": y_sim
    })

# Resposta teórica ao impulso calculada no vetor de tempo
t_h = np.linspace(0, t_max, 3000)
h_teorico = b * np.exp(-a * t_h)

# ==============================================================================
# 2. GERAÇÃO E EXPORTAÇÃO DAS TABELAS
# ==============================================================================
tabela_txt_path = os.path.join(OUTPUT_DIR, "tabela_letra_d.txt")
tabela_csv_path = os.path.join(OUTPUT_DIR, "tabela_letra_d.csv")

linhas_txt = []
linhas_txt.append("=" * 115)
linhas_txt.append("LABORATÓRIO DE SISTEMAS DE CONTROLE - QUESTÃO 2 - LETRA D")
linhas_txt.append(f"Parâmetros da Simulação 6: R1={R1:.1f} Ohm | R2={R2:.1f} Ohm | C={C:.2f} F | mu={mu:.1f} | R3={R3:.1f} Ohm")
linhas_txt.append(f"Modelo: a = {a:.5f} rad/s | b = {b:.5f} V/s | tau = {tau:.4f} s | K = {K:.4f} V")
linhas_txt.append(f"Resposta Ideal ao Impulso: h(t) = {b:.5f} * exp(-{a:.5f} * t)  ==>  h(0+) = b = {h0:.5f} V")
linhas_txt.append("=" * 115)
linhas_txt.append(f"{'Sim':>4} | {'L (s)':>8} | {'A = 1/L':>10} | {'L / tau':>10} | {'y(L) Pico (V)':>14} | {'h(0+) Alvo (V)':>14} | {'Erro %':>10} | {'Classificação':<30}")
linhas_txt.append("-" * 115)

linhas_csv = ["Simulacao,L_s,Amplitude_A,L_sobre_tau,y_pico_exato_V,y_pico_sim_V,h0_alvo_V,Erro_Rel_pct,y_taylor_1_V,y_taylor_2_V,Classificacao"]

for d in dados_processados:
    linha = (f"{d['sim']:4d} | {d['L']:8.4f} | {d['A']:10.4f} | {d['razao_L_tau']:10.4f} | "
             f"{d['y_pico_exato']:14.5f} | {h0:14.5f} | {d['erro_rel_h0']:9.2f}% | {d['classif']:<30}")
    linhas_txt.append(linha)
    
    linhas_csv.append(f"{d['sim']},{d['L']},{d['A']},{d['razao_L_tau']:.5f},{d['y_pico_exato']:.5f},"
                      f"{d['y_pico_sim']:.5f},{h0:.5f},{d['erro_rel_h0']:.3f},{d['y_taylor_1']:.5f},"
                      f"{d['y_taylor_2']:.5f},\"{d['classif']}\"")

linhas_txt.append("=" * 115)
linhas_txt.append("\nTABELA COMPLEMENTAR: VALIDAÇÃO DA APROXIMAÇÃO POR SÉRIE DE TAYLOR")
linhas_txt.append(f"{'Sim':>4} | {'L (s)':>8} | {'y(L) Exato (V)':>15} | {'Taylor 1ª Ordem':>16} | {'Taylor 2ª Ordem':>16} | {'Erro Taylor 1ª (%)':>20}")
linhas_txt.append("-" * 115)

for d in dados_processados:
    erro_taylor1 = abs(d["y_taylor_1"] - d["y_pico_exato"]) / d["y_pico_exato"] * 100.0
    linhas_txt.append(f"{d['sim']:4d} | {d['L']:8.4f} | {d['y_pico_exato']:15.5f} | {d['y_taylor_1']:16.5f} | {d['y_taylor_2']:16.5f} | {erro_taylor1:19.3f}%")

linhas_txt.append("=" * 115)
linhas_txt.append("\nCONCLUSÃO MATEMÁTICA E FÍSICA:")
linhas_txt.append("1. A resposta ao pulso converge rigorosamente para a resposta ao impulso quando L -> 0.")
linhas_txt.append(f"2. Para L = 0.0625 s (Simulação 5), L/tau = {0.0625/tau:.4f} << 1, e o erro frente ao impulso ideal é de apenas {dados_processados[-1]['erro_rel_h0']:.2f}%.")
linhas_txt.append("3. Condição Prática de Equivalência: L <= tau / 50 (ou seja, L <= 0.02 * tau).")

conteudo_txt = "\n".join(linhas_txt)
with open(tabela_txt_path, "w", encoding="utf-8") as f:
    f.write(conteudo_txt)

with open(tabela_csv_path, "w", encoding="utf-8") as f:
    f.write("\n".join(linhas_csv))

print(conteudo_txt)
print(f"\n[OK] Tabelas exportadas para:\n  -> {tabela_txt_path}\n  -> {tabela_csv_path}")

# ==============================================================================
# 3. GERAÇÃO DE GRÁFICOS
# ==============================================================================
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({'font.size': 10, 'figure.autolayout': True})

cores_pulso = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# --- GRÁFICO 1: COMPARATIVO GERAL MACROSCÓPICO (t = 0 a 80 s) ---
fig, ax = plt.subplots(figsize=(11, 6.5))

# Resposta ao impulso ideal h(t)
ax.plot(t_h, h_teorico, 'k--', lw=2.8, label=f'Impulso Ideal $h(t) = {b:.4f} \\cdot e^{{-{a:.4f} t}}$ ($\\delta(t)$)')

for idx, d in enumerate(dados_processados):
    ax.plot(d["t_sim"], d["y_sim"], color=cores_pulso[idx], lw=2.0,
            label=f'Sim {d["sim"]} ($L = {d["L"]:.4g}$ s, $A = {d["A"]:.3g}$): Pico={d["y_pico_exato"]:.4f} V')

ax.set_title('Convergência da Resposta ao Pulso $y_L(t)$ para a Resposta ao Impulso $h(t)$\n(Simulação 6: $\\mu = 2.0$, $R_3 = 200\\;\\Omega$, $\\tau = 16.45$ s)', fontsize=13, fontweight='bold')
ax.set_xlabel('Tempo $t$ (s)', fontsize=11)
ax.set_ylabel('Tensão de Saída $y(t)$ (V)', fontsize=11)
ax.set_xlim(0, 80)
ax.set_ylim(-0.005, 0.14)
ax.legend(loc='upper right', frameon=True, fontsize=9.5)
ax.grid(True, linestyle=':', alpha=0.6)

fig_path_comp = os.path.join(OUTPUT_DIR, "grafico_letra_d_comparativo.png")
plt.savefig(fig_path_comp, dpi=300, bbox_inches='tight')
plt.close()
print(f"[OK] Gráfico 1 salvo: {fig_path_comp}")

# --- GRÁFICO 2: ZOOM NO REGIME TRANSITÓRIO INICIAL (t = 0 a 5 s) ---
fig, ax = plt.subplots(figsize=(11, 6))

t_zoom = np.linspace(0, 5, 1000)
h_zoom = b * np.exp(-a * t_zoom)
ax.plot(t_zoom, h_zoom, 'k--', lw=2.8, label=f'Impulso Ideal $h(t)$ [Pico inicial $h(0^+) = {b:.4f}$ V]')

for idx, d in enumerate(dados_processados):
    # Filtrar dados para t <= 5s
    mask = d["t_sim"] <= 5.0
    ax.plot(d["t_sim"][mask], d["y_sim"][mask], color=cores_pulso[idx], lw=2.2,
            label=f'Sim {d["sim"]} ($L = {d["L"]:.4g}$ s) $\\to$ Pico = {d["y_pico_exato"]:.4f} V (Erro: {d["erro_rel_h0"]:.2f}%)')
    # Ponto no instante L se L <= 5s
    if d["L"] <= 5.0:
        ax.plot(d["L"], d["y_pico_exato"], 'o', color=cores_pulso[idx], markersize=6)

ax.axhline(h0, color='gray', linestyle=':', lw=1.2, alpha=0.8)
ax.set_title('Zoom no Regime Transitório Inicial ($t \\in [0, 5]$ s): Carga do Capacitor e Atingimento do Pico $y(L)$', fontsize=12, fontweight='bold')
ax.set_xlabel('Tempo $t$ (s)', fontsize=11)
ax.set_ylabel('Tensão de Saída $y(t)$ (V)', fontsize=11)
ax.set_xlim(0, 5)
ax.set_ylim(0, 0.135)
ax.legend(loc='lower right', frameon=True, fontsize=9)
ax.grid(True, linestyle=':', alpha=0.6)

fig_path_zoom = os.path.join(OUTPUT_DIR, "grafico_letra_d_zoom_inicial.png")
plt.savefig(fig_path_zoom, dpi=300, bbox_inches='tight')
plt.close()
print(f"[OK] Gráfico 2 salvo: {fig_path_zoom}")

# --- GRÁFICO 3: PAINEL COM AS 5 SIMULAÇÕES (ENTRADA u(t) E SAÍDA y(t)) ---
fig, axes = plt.subplots(5, 1, figsize=(11, 14), sharex=True)

for idx, d in enumerate(dados_processados):
    ax = axes[idx]
    
    # Eixo principal: Saída y(t)
    cor_y = cores_pulso[idx]
    ax.plot(t_h, h_teorico, 'k--', lw=1.5, alpha=0.7, label='Impulso Ideal $h(t)$')
    ax.plot(d["t_sim"], d["y_sim"], color=cor_y, lw=2.2, label=f'Saída $y(t)$ (Sim {d["sim"]}: $L={d["L"]:.4g}$ s)')
    ax.plot(d["L"], d["y_pico_exato"], 'ro', markersize=5, label=f'Pico $y(L) = {d["y_pico_exato"]:.4f}$ V')
    ax.set_ylabel('Saída $y(t)$ (V)', color='navy', fontsize=10)
    ax.set_ylim(-0.01, 0.14)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Eixo secundário: Entrada u(t)
    ax_u = ax.twinx()
    ax_u.plot(d["t_sim"], d["u_sim"], color='gray', linestyle='-', lw=1.2, alpha=0.8, label=f'Entrada $u(t)$ ($A = {d["A"]:.3g}$ V)')
    ax_u.set_ylabel('Entrada $u(t)$ (V)', color='gray', fontsize=9)
    ax_u.tick_params(axis='y', labelcolor='gray')
    
    # Limitar eixo de entrada para boa visualização
    ax_u.set_ylim(0, d["A"] * 1.25)
    
    titulo = f'Simulação {d["sim"]}: $L = {d["L"]}$ s, $A = {d["A"]:.4g}$ V, $L/\\tau = {d["razao_L_tau"]:.4f}$ | Erro frente a $h(0^+)$: {d["erro_rel_h0"]:.2f}%'
    ax.set_title(titulo, fontsize=10.5, fontweight='bold')
    
    # Combinar legendas
    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax_u.get_legend_handles_labels()
    ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', fontsize=8)

axes[-1].set_xlabel('Tempo $t$ (s)', fontsize=11)
axes[-1].set_xlim(0, 80)

fig_path_paineis = os.path.join(OUTPUT_DIR, "grafico_letra_d_entradas_e_saidas.png")
plt.savefig(fig_path_paineis, dpi=300, bbox_inches='tight')
plt.close()
print(f"[OK] Gráfico 3 salvo: {fig_path_paineis}")

# --- GRÁFICO 4: ERRO RELATIVO E CONVERGÊNCIA DE TAYLOR ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

L_valores = [d["L"] for d in dados_processados]
erros_valores = [d["erro_rel_h0"] for d in dados_processados]
picos_valores = [d["y_pico_exato"] for d in dados_processados]
taylor1_valores = [d["y_taylor_1"] for d in dados_processados]

# Subplot 1: Pico y(L) vs h(0+) e aproximação de Taylor
L_continuo = np.linspace(0.01, 35, 500)
y_pico_continuo = (K / L_continuo) * (1.0 - np.exp(-L_continuo / tau))
taylor1_continuo = b * (1.0 - L_continuo / (2.0 * tau))

ax1.plot(L_continuo, y_pico_continuo, 'b-', lw=2.2, label='Pico Exato $y(L) = \\frac{K}{L}(1 - e^{-L/\\tau})$')
ax1.plot(L_continuo, taylor1_continuo, 'r--', lw=1.8, label=r'Taylor 1ª Ordem: $b(1 - \frac{L}{2\tau})$')
ax1.axhline(h0, color='k', linestyle=':', lw=2, label=f'Impulso Alvo $h(0^+) = b = {h0:.4f}$ V')
ax1.plot(L_valores, picos_valores, 'go', markersize=7, label='Pontos das 5 Simulações')
ax1.set_title('Valor de Pico $y(L)$ em Função da Duração do Pulso $L$', fontsize=11, fontweight='bold')
ax1.set_xlabel('Largura do Pulso $L$ (s)', fontsize=10)
ax1.set_ylabel('Pico $y(L)$ (V)', fontsize=10)
ax1.set_xlim(0, 35)
ax1.legend(loc='lower left', frameon=True, fontsize=9)
ax1.grid(True, linestyle=':', alpha=0.6)

# Subplot 2: Erro Relativo (%) vs L/tau
razoes_L = np.array([d["razao_L_tau"] for d in dados_processados])
razao_cont = np.linspace(0.001, 2.0, 500)
erro_exato_cont = (1.0 - (1.0 - np.exp(-razao_cont)) / razao_cont) * 100.0
erro_taylor_cont = (razao_cont / 2.0) * 100.0

ax2.plot(razao_cont, erro_exato_cont, 'b-', lw=2.2, label='Erro Relativo Exato (%)')
ax2.plot(razao_cont, erro_taylor_cont, 'r--', lw=1.8, label='Aproximação de Taylor: $\\frac{L}{2\\tau} \\times 100\\%$')
ax2.plot(razoes_L, erros_valores, 'go', markersize=7, label='Simulações 1 a 5')
ax2.set_title('Erro Relativo em Relação ao Impulso vs $L / \\tau$', fontsize=11, fontweight='bold')
ax2.set_xlabel('Razão $L / \\tau$', fontsize=10)
ax2.set_ylabel('Erro Relativo (%)', fontsize=10)
ax2.set_xlim(0, 2.0)
ax2.set_ylim(0, 60)
ax2.legend(loc='upper left', frameon=True, fontsize=9)
ax2.grid(True, linestyle=':', alpha=0.6)

fig_path_taylor = os.path.join(OUTPUT_DIR, "grafico_letra_d_convergencia_taylor.png")
plt.savefig(fig_path_taylor, dpi=300, bbox_inches='tight')
plt.close()
print(f"[OK] Gráfico 4 salvo: {fig_path_taylor}")

print("\n>>> Simulação da Letra (d) concluída com sucesso! <<<")
