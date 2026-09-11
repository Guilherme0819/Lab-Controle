#!/usr/bin/env python3
"""
Laboratório de Sistemas de Controle - UFAM
Experimento 1: Comportamento dinâmico de sistemas de 1ª e 2ª ordem
Questão 2 - Letra (c): Resposta ao Degrau Unitário

Este script realiza a simulação computacional das 6 configurações solicitadas,
calcula os parâmetros teóricos de controle (polos, constante de tempo, ganho estático,
tempos de acomodação), simula as respostas dinâmicas com a biblioteca 'control',
gera gráficos comparativos e individuais em alta resolução, e exporta tabelas de dados
formatadas para fácil transferência para o relatório.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import control as ct

# Diretório base para salvar figuras e tabelas
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Parâmetros fixos do circuito
R1 = 10.0   # Ohms
R2 = 50.0   # Ohms
C  = 0.25   # Farads

# Tabela de simulações fornecida no enunciado
# Simulação | mu  | R3 (Ohms)
simulacoes_info = [
    {"sim": 1, "mu": 0.5, "R3": 5.0},
    {"sim": 2, "mu": 0.5, "R3": 10.0},
    {"sim": 3, "mu": 0.5, "R3": 200.0},
    {"sim": 4, "mu": 2.0, "R3": 5.0},
    {"sim": 5, "mu": 2.0, "R3": 10.0},
    {"sim": 6, "mu": 2.0, "R3": 200.0},
]

def calcular_parametros(R1, R2, R3, C, mu):
    """
    Calcula os coeficientes da EDO: dy/dt + a*y = b*u(t)
    Função de transferência: G(s) = b / (s + a)
    """
    denom = (R1 * R2 + R2 * R3 + R3 * R1) * C
    num_a = (1.0 - mu) * R1 + R3
    num_b = mu * R3

    a = num_a / denom
    b = num_b / denom
    polo = -a

    if abs(a) < 1e-9:
        # Integrador puro (polo na origem)
        tau = np.inf
        K = np.nan
        ts_2pct = np.inf
        ts_1pct = np.inf
        classificacao = "Integrador Puro (Marginalmente Estável - Polo em s=0)"
    elif a < 0:
        # Instável (polo no semiplano direito)
        tau = 1.0 / a  # negativo
        K = b / a      # ganho formal
        ts_2pct = np.nan
        ts_1pct = np.nan
        classificacao = "Instável (Polo no SPD s > 0)"
    else:
        # Estável de 1ª ordem
        tau = 1.0 / a
        K = b / a
        ts_2pct = 4.0 * tau
        ts_1pct = 5.0 * tau
        classificacao = "Estável (1ª Ordem)"

    return {
        "denom": denom,
        "a": a,
        "b": b,
        "polo": polo,
        "tau": tau,
        "K": K,
        "ts_2pct": ts_2pct,
        "ts_1pct": ts_1pct,
        "classificacao": classificacao
    }

# Processamento de todos os casos
resultados = []
for sim in simulacoes_info:
    p = calcular_parametros(R1, R2, sim["R3"], C, sim["mu"])
    item = {**sim, **p}
    # Criar sistema no control
    # G(s) = b / (s + a)
    sys = ct.tf([p["b"]], [1.0, p["a"]])
    item["sys"] = sys
    resultados.append(item)

# ==============================================================================
# 1. GERAÇÃO DAS TABELAS FORMATADAS
# ==============================================================================
tabela_txt_path = os.path.join(OUTPUT_DIR, "tabela_letra_c.txt")
tabela_csv_path = os.path.join(OUTPUT_DIR, "tabela_letra_c.csv")

linhas_txt = []
linhas_txt.append("=" * 105)
linhas_txt.append("LABORATÓRIO DE SISTEMAS DE CONTROLE - QUESTÃO 2 - LETRA C")
linhas_txt.append(f"Parâmetros Fixos: R1 = {R1:.1f} Ohm | R2 = {R2:.1f} Ohm | C = {C:.2f} F")
linhas_txt.append("Equação Diferencial: dy/dt + a*y(t) = b*u(t)  ==>  G(s) = b / (s + a)")
linhas_txt.append("=" * 105)
linhas_txt.append(f"{'Sim':>4} | {'mu':>4} | {'R3 (Ohm)':>8} | {'Polo (rad/s)':>12} | {'tau (s)':>10} | {'K (V/V)':>10} | {'b (V/s)':>10} | {'ts 2% (s)':>10} | {'Classificação':<25}")
linhas_txt.append("-" * 105)

linhas_csv = ["Simulacao,mu,R3_Ohm,a_rad_s,b_V_s,Polo_rad_s,tau_s,K_regime,ts_2pct_s,ts_1pct_s,Classificacao"]

for r in resultados:
    tau_str = f"{r['tau']:10.4f}" if np.isfinite(r['tau']) and r['tau'] > 0 else ("inf" if np.isinf(r['tau']) else f"{r['tau']:10.4f} (SPD)")
    K_str = f"{r['K']:10.4f}" if np.isfinite(r['K']) else "inf/rampa"
    ts_str = f"{r['ts_2pct']:10.2f}" if np.isfinite(r['ts_2pct']) else "N/A"
    
    linha_formatada = (f"{r['sim']:4d} | {r['mu']:4.1f} | {r['R3']:8.1f} | "
                       f"{r['polo']:12.5f} | {tau_str} | {K_str} | {r['b']:10.5f} | "
                       f"{ts_str} | {r['classificacao']:<25}")
    linhas_txt.append(linha_formatada)
    
    linhas_csv.append(f"{r['sim']},{r['mu']},{r['R3']},{r['a']:.5f},{r['b']:.5f},{r['polo']:.5f},"
                      f"{r['tau']},{r['K']},{r['ts_2pct']},{r['ts_1pct']},\"{r['classificacao']}\"")

linhas_txt.append("=" * 105)
linhas_txt.append("\nANÁLISE DOS CASOS LIMITES:")
linhas_txt.append("1) Para R3 = 0 Ohm:")
linhas_txt.append("   - Denominador = (R1*R2)*C = 125.0")
linhas_txt.append("   - Numerador de b = mu*R3 = 0  ==>  b = 0")
linhas_txt.append("   - Numerador de a = (1 - mu)*R1")
linhas_txt.append("   - Conclusão Física: A entrada u(t) fica desacoplada da saída y(t).")
linhas_txt.append("     A saída permanece em y(t) = 0 para qualquer entrada u(t).")
linhas_txt.append("\n2) Para R3 -> infinito (R3 aberto):")
linhas_txt.append("   - a = 1 / ((R1 + R2)*C) = 1 / (60 * 0.25) = 1 / 15 s = 0.06667 rad/s")
linhas_txt.append("   - tau = (R1 + R2)*C = 15.0 s")
linhas_txt.append("   - b = mu / ((R1 + R2)*C)")
linhas_txt.append("   - K = mu (Ganho estático é exatamente igual a mu)")
linhas_txt.append("   - Conclusão Física: Circuito RC passivo série convencional com ganho estático mu.")

conteudo_txt = "\n".join(linhas_txt)
with open(tabela_txt_path, "w", encoding="utf-8") as f:
    f.write(conteudo_txt)

with open(tabela_csv_path, "w", encoding="utf-8") as f:
    f.write("\n".join(linhas_csv))

print(conteudo_txt)
print(f"\n[OK] Tabelas exportadas para:\n  -> {tabela_txt_path}\n  -> {tabela_csv_path}")

# ==============================================================================
# 2. SIMULAÇÃO E GERAÇÃO DE GRÁFICOS
# ==============================================================================
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({'font.size': 10, 'figure.autolayout': True})

# --- GRÁFICO 1: PAINEL COM AS 6 SIMULAÇÕES (2x3) ---
fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=False)
axes = axes.flatten()

t_comum = np.linspace(0, 100, 1000)

for idx, r in enumerate(resultados):
    ax = axes[idx]
    sim_id = r["sim"]
    
    if r["polo"] > 0:
        # Sistema instável: limitar tempo para não estourar escala
        t_sim = np.linspace(0, 80, 800)
        # Solução analítica exata: y(t) = (b/a)*(1 - exp(-a*t)) = K*(1 - exp(-a*t))
        # Para a = -0.025: y(t) = -2 * (1 - exp(0.025*t)) = 2*(exp(0.025*t) - 1)
        y_sim = r["K"] * (1.0 - np.exp(-r["a"] * t_sim))
        ax.plot(t_sim, y_sim, color='crimson', lw=2.5, label=r'$y(t)$ simulado (Instável)')
        ax.set_title(f'Simulação {sim_id}: $\\mu = {r["mu"]}$, $R_3 = {r["R3"]}\\;\\Omega$\n[INSTÁVEL - Polo no SPD: $s = +{r["polo"]:.4f}$]', color='crimson', fontweight='bold')
        ax.set_ylabel('Saída $y(t)$ (V)')
        ax.set_xlabel('Tempo $t$ (s)')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.6)
        ax.legend(loc='upper left')
        
    elif abs(r["a"]) < 1e-9:
        # Integrador puro: y(t) = b * t
        t_sim = np.linspace(0, 100, 1000)
        y_sim = r["b"] * t_sim
        ax.plot(t_sim, y_sim, color='darkorange', lw=2.5, label=f'$y(t) = {r["b"]:.4f} \\cdot t$ (Rampa)')
        ax.set_title(f'Simulação {sim_id}: $\\mu = {r["mu"]}$, $R_3 = {r["R3"]}\\;\\Omega$\n[INTEGRADOR PURO - Polo na Origem: $s = 0$]', color='darkorange', fontweight='bold')
        ax.set_ylabel('Saída $y(t)$ (V)')
        ax.set_xlabel('Tempo $t$ (s)')
        ax.legend(loc='upper left')
        
    else:
        # Sistema estável
        t_sim = np.linspace(0, 5 * r["tau"], 1000)
        t_out, y_out = ct.step_response(r["sys"], t_sim)
        ax.plot(t_out, y_out, color='navy', lw=2.5, label=r'$y(t)$ simulado')
        
        # Linha de regime permanente K
        ax.axhline(r["K"], color='darkgreen', linestyle='--', lw=1.8, label=f'Regime K = {r["K"]:.4f} V')
        
        # Ponto em t = tau (63.2%)
        y_tau = r["K"] * (1.0 - np.exp(-1.0))
        ax.plot(r["tau"], y_tau, 'ro', markersize=7, label=f'$\\tau = {r["tau"]:.2f}$ s $\\to$ 63.2% K')
        ax.vlines(r["tau"], 0, y_tau, color='gray', linestyle=':', alpha=0.7)
        
        # Reta tangente na origem (taxa de crescimento inicial)
        t_tangente = np.linspace(0, min(r["tau"], 25.0), 100)
        y_tangente = r["b"] * t_tangente
        ax.plot(t_tangente, y_tangente, color='purple', linestyle='-.', lw=1.2, label=f'Tangente inicial (b = {r["b"]:.4f})')
        
        ax.set_title(f'Simulação {sim_id}: $\\mu = {r["mu"]}$, $R_3 = {r["R3"]}\\;\\Omega$\n[ESTÁVEL: $\\tau = {r["tau"]:.2f}$ s, $K = {r["K"]:.4f}$ V]', fontweight='bold')
        ax.set_ylabel('Saída $y(t)$ (V)')
        ax.set_xlabel('Tempo $t$ (s)')
        ax.set_ylim(-0.05 * r["K"], 1.15 * r["K"])
        ax.legend(loc='lower right', fontsize=8)

fig_path_todas = os.path.join(OUTPUT_DIR, "grafico_letra_c_todas.png")
plt.savefig(fig_path_todas, dpi=300, bbox_inches='tight')
plt.close()
print(f"[OK] Gráfico 1 salvo: {fig_path_todas}")

# --- GRÁFICO 2: COMPARATIVO DAS 4 SIMULAÇÕES ESTÁVEIS (Sim 1, 2, 3 e 6) ---
fig, ax = plt.subplots(figsize=(10, 6))
t_comp = np.linspace(0, 100, 1000)
cores = ['#1f77b4', '#2ca02c', '#9467bd', '#d62728']

for idx, sim_id in enumerate([1, 2, 3, 6]):
    r = resultados[sim_id - 1]
    t_out, y_out = ct.step_response(r["sys"], t_comp)
    ax.plot(t_out, y_out, color=cores[idx], lw=2.2, 
            label=f'Sim {sim_id}: $\\mu={r["mu"]}$, $R_3={r["R3"]}\\;\\Omega$ ($\\tau={r["tau"]:.1f}$ s, $K={r["K"]:.3f}$ V)')
    # Linha tracejada do regime
    ax.axhline(r["K"], color=cores[idx], linestyle='--', alpha=0.5, lw=1.2)

ax.set_title('Comparativo de Resposta ao Degrau - Casos Estáveis (Simulações 1, 2, 3 e 6)', fontsize=13, fontweight='bold')
ax.set_xlabel('Tempo $t$ (s)', fontsize=11)
ax.set_ylabel('Tensão de Saída $y(t)$ (V)', fontsize=11)
ax.set_xlim(0, 100)
ax.legend(loc='upper left', frameon=True, fontsize=9.5)
ax.grid(True, linestyle=':', alpha=0.6)

fig_path_comp = os.path.join(OUTPUT_DIR, "grafico_letra_c_comparativo_estaveis.png")
plt.savefig(fig_path_comp, dpi=300, bbox_inches='tight')
plt.close()
print(f"[OK] Gráfico 2 salvo: {fig_path_comp}")

# --- GRÁFICO 3: EFEITO DE BIFURCAÇÃO DE ESTABILIDADE COM mu = 2.0 (Sim 4, 5 e 6) ---
fig, ax = plt.subplots(figsize=(10, 6))
t_bif = np.linspace(0, 60, 600)

# Sim 4 (Instável)
r4 = resultados[3]
y4 = r4["K"] * (1.0 - np.exp(-r4["a"] * t_bif))
ax.plot(t_bif, y4, 'r-', lw=2.5, label=f'Sim 4 ($R_3=5\\;\\Omega$): Instável ($s = +{r4["polo"]:.4f}$, crescimento exponencial)')

# Sim 5 (Integrador)
r5 = resultados[4]
y5 = r5["b"] * t_bif
ax.plot(t_bif, y5, 'orange', lw=2.5, linestyle='--', label=f'Sim 5 ($R_3=10\\;\\Omega$): Integrador Puro ($s = 0$, rampa $y=bt$)')

# Sim 6 (Estável)
r6 = resultados[5]
t_out6, y_out6 = ct.step_response(r6["sys"], t_bif)
ax.plot(t_out6, y_out6, 'b-', lw=2.5, label=f'Sim 6 ($R_3=200\\;\\Omega$): Estável ($s = -{r6["a"]:.4f}$, $\\tau=16.45$ s, $K=2.105$ V)')
ax.axhline(r6["K"], color='b', linestyle=':', alpha=0.7, label=f'Regime Estável K = {r6["K"]:.3f} V')

ax.set_title('Efeito do Resistor de Realimentação $R_3$ sobre a Estabilidade (para $\\mu = 2.0$)', fontsize=13, fontweight='bold')
ax.set_xlabel('Tempo $t$ (s)', fontsize=11)
ax.set_ylabel('Tensão de Saída $y(t)$ (V)', fontsize=11)
ax.set_ylim(-0.5, 10.0)
ax.set_xlim(0, 60)
ax.legend(loc='upper left', frameon=True, fontsize=9.5)
ax.grid(True, linestyle=':', alpha=0.6)

fig_path_bif = os.path.join(OUTPUT_DIR, "grafico_letra_c_mu2_estabilidade.png")
plt.savefig(fig_path_bif, dpi=300, bbox_inches='tight')
plt.close()
print(f"[OK] Gráfico 3 salvo: {fig_path_bif}")

# --- GRÁFICO 4: CASOS LIMITES (R3 = 0 e R3 -> infinito) ---
fig, ax = plt.subplots(figsize=(10, 5.5))
t_lim = np.linspace(0, 80, 800)

# R3 = 0: y(t) = 0
y_r3_0 = np.zeros_like(t_lim)
ax.plot(t_lim, y_r3_0, 'k--', lw=2.5, label=r'$R_3 = 0\,\Omega$: Saída nula $y(t) = 0$ (Entrada desacoplada)')

# R3 -> infinito para mu = 0.5: tau = 15s, K = 0.5
tau_inf = (R1 + R2) * C # 15s
K_inf_05 = 0.5
y_inf_05 = K_inf_05 * (1.0 - np.exp(-t_lim / tau_inf))
ax.plot(t_lim, y_inf_05, color='teal', lw=2.2, label=f'$R_3 \\to \\infty$ com $\\mu = 0.5$: $\\tau = 15.0$ s, $K = 0.50$ V')

# R3 -> infinito para mu = 2.0: tau = 15s, K = 2.0
K_inf_20 = 2.0
y_inf_20 = K_inf_20 * (1.0 - np.exp(-t_lim / tau_inf))
ax.plot(t_lim, y_inf_20, color='darkmagenta', lw=2.2, label=f'$R_3 \\to \\infty$ com $\\mu = 2.0$: $\\tau = 15.0$ s, $K = 2.00$ V')

ax.set_title('Casos Limites: Resposta ao Degrau para $R_3 = 0\\;\\Omega$ e $R_3 \\to \\infty$ (Aberto)', fontsize=13, fontweight='bold')
ax.set_xlabel('Tempo $t$ (s)', fontsize=11)
ax.set_ylabel('Tensão de Saída $y(t)$ (V)', fontsize=11)
ax.set_xlim(0, 80)
ax.legend(loc='center right', frameon=True, fontsize=10)
ax.grid(True, linestyle=':', alpha=0.6)

fig_path_lim = os.path.join(OUTPUT_DIR, "grafico_letra_c_casos_limites.png")
plt.savefig(fig_path_lim, dpi=300, bbox_inches='tight')
plt.close()
print(f"[OK] Gráfico 4 salvo: {fig_path_lim}")

print("\n>>> Simulação da Letra (c) concluída com sucesso! <<<")
