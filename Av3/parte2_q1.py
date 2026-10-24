"""
Script de Verificação Numérica - Laboratório 03: Sistemas de Controle (UFAM)
Experimento 03 - Parte 02: Questão 1 (Modelagem do Sistema em Espaço de Estados)
Executa com python3 padrão (módulo math) e valida matrizes, funções de transferência,
polos, frações parciais e valores de regime permanente.
"""

import math

# ==========================================
# 1. PARÂMETROS FORNECIDOS NO ROTEIRO (PARTE 2)
# ==========================================
Ra = 10.0          # Resistência de armadura [Ohms]
La = 125e-3        # Indutância de armadura [H] (125 mH)
Kb = 0.45          # Constante de fcem [V.s/rad]
KT = 15.0          # Constante de torque [N.m/A]

Jm = 2.5           # Inércia rotor + engr 1 [kg.m^2]
J2 = 0.075         # Inércia engr 2 + engr 3 [kg.m^2]
J4 = 20.0          # Inércia engr 4 + antena [kg.m^2]

B1 = 0.25          # Coef. atrito mancal rotor [N.m.s/rad] (Bm)
Bm = B1
B2 = 0.085         # Coef. atrito engrs. 2 e 3 [N.m.s/rad]
B4 = 0.35          # Coef. atrito engr 4 + antena [N.m.s/rad]

N1 = 16            # Dentes engr 1 (eixo motor)
N2 = 100           # Dentes engr 2 (eixo intermediário)
N3 = 40            # Dentes engr 3 (eixo intermediário)
N4 = 160           # Dentes engr 4 (eixo antena/carga)

# ==========================================
# 2. RELAÇÕES DE TRANSMISSÃO
# ==========================================
n12 = N1 / N2                 # Estágio 1: 16/100 = 0.16 = 4/25
n34 = N3 / N4                 # Estágio 2: 40/160 = 0.25 = 1/4
ntot = (N1 * N3) / (N2 * N4)  # Relação total: 16*40 / (100*160) = 640/16000 = 0.04 = 1/25

# ==========================================
# 3. CÁLCULO DE Jeq E Beq VISTOS PELO MOTOR
# ==========================================
J2_refletido = (n12**2) * J2
J4_refletido = (ntot**2) * J4
Jeq = Jm + J2_refletido + J4_refletido

B2_refletido = (n12**2) * B2
B4_refletido = (ntot**2) * B4
Beq = Bm + B2_refletido + B4_refletido

# ==========================================
# 4. MATRIZES DE ESPAÇO DE ESTADOS (ITEM 1.a)
# x = [i_a, omega_m, theta_L]^T, u = e_a, y = theta_L
# ==========================================
A11 = -Ra / La
A12 = -Kb / La
A13 = 0.0

A21 = KT / Jeq
A22 = -Beq / Jeq
A23 = 0.0

A31 = 0.0
A32 = ntot
A33 = 0.0

B1_vec = 1.0 / La
B2_vec = 0.0
B3_vec = 0.0

# ==========================================
# 5. FUNÇÃO DE TRANSFERÊNCIA Gm(s) = Omega_m(s) / Ea(s)
# ==========================================
num_Gm = KT / (La * Jeq)
den_s2 = 1.0
den_s1 = (Ra / La) + (Beq / Jeq)
den_s0 = (Ra * Beq + KT * Kb) / (La * Jeq)

delta = den_s1**2 - 4 * den_s2 * den_s0
p1 = (-den_s1 + math.sqrt(delta)) / 2.0  # Polo mecânico (dominante)
p2 = (-den_s1 - math.sqrt(delta)) / 2.0  # Polo elétrico (rápido)

# ==========================================
# 6. RESPOSTA AO DEGRAU ea(t) = 110 V (ITEM 1.b)
# ==========================================
Ea_step = 110.0

# Omega_m(s) = Ea_step * num_Gm / (s * (s - p1) * (s - p2))
# Frações parciais: A/s + B/(s - p1) + C/(s - p2)
res_A = (Ea_step * num_Gm) / (p1 * p2)
res_B = (Ea_step * num_Gm) / (p1 * (p1 - p2))
res_C = (Ea_step * num_Gm) / (p2 * (p2 - p1))

omega_inf_rad_s = res_A
omega_inf_rpm = omega_inf_rad_s * 60.0 / (2.0 * math.pi)

# Posicao da Antena: Theta_L(s) = (ntot / s) * Omega_m(s)
# Resposta em regime permanente da velocidade da carga:
omega_L_inf = ntot * omega_inf_rad_s
omega_L_inf_deg_s = omega_L_inf * 180.0 / math.pi

# Constantes de tempo e tempo de acomodação (2%):
tau1 = 1.0 / abs(p1)
tau2 = 1.0 / abs(p2)
ts_2 = 4.0 * tau1

if __name__ == "__main__":
    print("=" * 75)
    print("  LABORATÓRIO 03 - EXPERIMENTO 03 (PARTE 02) - QUESTÃO 1")
    print("=" * 75)
    print("\n1. Relações de Transmissão Mecânica:")
    print(f"   - Estágio 1 (n_12 = N1/N2): {N1}/{N2} = {n12:.4f} = 4/25")
    print(f"   - Estágio 2 (n_34 = N3/N4): {N3}/{N4} = {n34:.4f} = 1/4")
    print(f"   - Total (n_tot = (N1*N3)/(N2*N4)): ({N1}*{N3})/({N2}*{N4}) = {ntot:.4f} = 1/25")

    print("\n2. Inércia e Atrito Equivalentes (vistos pelo motor):")
    print(f"   - Jm (rotor + engr 1):       {Jm:.6f} kg.m^2")
    print(f"   - J2 refletido (n_12^2 * J2): {J2_refletido:.8f} kg.m^2")
    print(f"   - J4 refletido (n_tot^2 * J4):{J4_refletido:.8f} kg.m^2")
    print(f"   >> Jeq Calculado:            {Jeq:.6f} kg.m^2 (exato: 2.53392)")
    print(f"   >> Jeq Roteiro (com erro):   2.5003 kg.m^2")
    print(f"   - Bm (rotor + engr 1):       {Bm:.6f} N.m.s/rad")
    print(f"   - B2 refletido (n_12^2 * B2): {B2_refletido:.8f} N.m.s/rad")
    print(f"   - B4 refletido (n_tot^2 * B4):{B4_refletido:.8f} N.m.s/rad")
    print(f"   >> Beq Calculado:            {Beq:.6f} N.m.s/rad (exato: 0.252736)")
    print(f"   >> Beq Roteiro (com erro):   0.2500 N.m.s/rad")
    print("   * Veredito: Conforme autorização do Prof. Cláudio Duarte, adotam-se os valores calculados.")

    print("\n3. Matrizes de Espaço de Estados (Item 1.a):")
    print(f"   A = [ [{A11:10.4f}, {A12:10.4f}, {A13:10.4f}]")
    print(f"         [{A21:10.4f}, {A22:10.4f}, {A23:10.4f}]")
    print(f"         [{A31:10.4f}, {A32:10.4f}, {A33:10.4f}] ]")
    print(f"   B = [ [{B1_vec:10.4f}], [{B2_vec:10.4f}], [{B3_vec:10.4f}] ]")
    print("   C = [ [0.0, 0.0, 1.0] ]")
    print("   D = [ [0.0] ]")

    print("\n4. Função de Transferência do Motor Gm(s) = Omega_m(s) / Ea(s):")
    print(f"   Gm(s) = {num_Gm:.6f} / ( s^2 + {den_s1:.6f} s + {den_s0:.6f} )")
    print("   Polos de Gm(s):")
    print(f"     p1 = {p1:.6f} rad/s (Polo Mecânico Dominante, tau1 = {tau1:.4f} s)")
    print(f"     p2 = {p2:.6f} rad/s (Polo Elétrico Rápido, tau2 = {tau2*1000:.2f} ms)")
    print(f"   Razão de Dominância |p2/p1| = {abs(p2/p1):.2f}x (Fortemente dominado por p1)")

    print("\n5. Resposta ao Degrau ea(t) = 110 V (Item 1.b):")
    print(f"   omega_m(t) = {res_A:.4f} + ({res_B:.4f})*exp({p1:.4f}*t) + ({res_C:.4f})*exp({p2:.4f}*t)  [rad/s]")
    print(f"   N(t)       = {res_A*60/(2*math.pi):.2f} + ({res_B*60/(2*math.pi):.2f})*exp({p1:.4f}*t) + ({res_C*60/(2*math.pi):.2f})*exp({p2:.4f}*t)  [RPM]")
    print(f"\n   Valores de Regime Permanente (t -> inf):")
    print(f"   - Velocidade do Motor em Regime: {omega_inf_rad_s:.4f} rad/s = {omega_inf_rpm:.2f} RPM")
    print(f"   - Velocidade da Antena em Regime: {omega_L_inf:.4f} rad/s = {omega_L_inf_deg_s:.2f} graus/s")
    print(f"   - Posição da Antena theta_L(t): Cresce indefinidamente como rampa linear com inclinação {omega_L_inf:.4f} rad/s ({omega_L_inf_deg_s:.2f} deg/s)")
    print(f"   - Tempo de Acomodação (critério de 2%): ts ≈ {ts_2:.2f} s")
    print("=" * 75)
