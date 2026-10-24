"""
Script de Verificação Numérica - Laboratório 03: Sistemas de Controle (UFAM)
Experimento 03 - Parte 01: Questão 1 (Modelagem do Sistema)
Executa com python3 padrão (módulo math) e com numpy/control se disponíveis.
"""

import math

# ==========================================
# 1. PARÂMETROS FORNECIDOS NO ROTEIRO (PARTE 1)
# ==========================================
Ra = 10.0          # Resistência de armadura [Ohms]
La = 125e-3        # Indutância de armadura [H] (125 mH)
Kb = 0.45          # Constante de fcem [V.s/rad]
KT = 15.0          # Constante de torque [N.m/A]

Jm = 2.5           # Inércia rotor + engr 1 [kg.m^2]
J2 = 0.075         # Inércia engr 2 + engr 3 [kg.m^2]
J4 = 20.0          # Inércia engr 4 + antena [kg.m^2]

Bm = 0.25          # Atrito rotor + engr 1 [N.m.s/rad]
B2 = 0.085         # Atrito engr 2 + engr 3 [N.m.s/rad]
B4 = 0.35          # Atrito engr 4 + antena [N.m.s/rad]

N1 = 15            # Dentes engr 1 (eixo motor)
N2 = 90            # Dentes engr 2 (eixo intermediário)
N3 = 30            # Dentes engr 3 (eixo intermediário)
N4 = 160           # Dentes engr 4 (eixo antena/carga)

# ==========================================
# 2. RELAÇÕES DE TRANSMISSÃO
# ==========================================
n12 = N1 / N2                 # Estágio 1: 15/90 = 1/6
n34 = N3 / N4                 # Estágio 2: 30/160 = 3/16
ntot = (N1 * N3) / (N2 * N4)  # Relação total: 15*30 / (90*160) = 450/14400 = 1/32

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
# 4. TRAMITÂNCIAS REALIZÁVEIS
# ==========================================
ganho_int_el = 1.0 / La
polo_el = Ra / La

ganho_int_mec = 1.0 / Jeq
polo_mec = Beq / Jeq

# ==========================================
# 5. FUNÇÃO DE TRANSFERÊNCIA EM MALHA ABERTA
# ==========================================
# Gm(s) = Omega_m(s) / Ea(s)
num_Gm = KT * ganho_int_el * ganho_int_mec
den_Gm_s2 = 1.0
den_Gm_s1 = polo_el + polo_mec
den_Gm_s0 = (polo_el * polo_mec) + (KT * Kb * ganho_int_el * ganho_int_mec)

# Raízes da equação característica: s^2 + a1*s + a0 = 0
delta = den_Gm_s1**2 - 4 * den_Gm_s2 * den_Gm_s0
if delta >= 0:
    p1 = (-den_Gm_s1 + math.sqrt(delta)) / (2 * den_Gm_s2)
    p2 = (-den_Gm_s1 - math.sqrt(delta)) / (2 * den_Gm_s2)
    polos_str = f"p1 = {p1:.6f} rad/s (Polo Dominante)\n     p2 = {p2:.6f} rad/s (Polo Rápido)"
else:
    real_part = -den_Gm_s1 / (2 * den_Gm_s2)
    imag_part = math.sqrt(-delta) / (2 * den_Gm_s2)
    polos_str = f"p1,2 = {real_part:.6f} +/- {imag_part:.6f}j"

if __name__ == "__main__":
    print("=" * 70)
    print("  LABORATÓRIO 03 - EXPERIMENTO 01 (PARTE 1) - QUESTÃO 1")
    print("=" * 70)
    print("\n1. Relações de Transmissão:")
    print(f"   - Estágio 1 (n_12 = N1/N2): {N1}/{N2} = {n12:.6f} = 1/{1/n12:.1f}")
    print(f"   - Estágio 2 (n_34 = N3/N4): {N3}/{N4} = {n34:.6f} = 3/16")
    print(f"   - Relação Total (n_tot):   {N1*N3}/{N2*N4} = {ntot:.6f} = 1/{1/ntot:.1f}")

    print("\n2. Inércia Equivalente (Jeq) Visto pelo Motor:")
    print(f"   - Jm (rotor + engr 1):       {Jm:.6f} kg.m^2")
    print(f"   - J2 refletido (n_12^2 * J2): {J2_refletido:.8f} kg.m^2")
    print(f"   - J4 refletido (n_tot^2 * J4):{J4_refletido:.8f} kg.m^2")
    print(f"   >> Jeq Calculado:            {Jeq:.6f} kg.m^2 (≈ 2.5216)")
    print(f"   >> Jeq Nominal do PDF:       2.5542 kg.m^2")
    print(f"   >> Veredito: Conforme autorização do Prof. Cláudio Duarte, adota-se 2.5216 kg.m^2.")

    print("\n3. Coeficiente de Atrito Equivalente (Beq) Visto pelo Motor:")
    print(f"   - Bm (rotor + engr 1):       {Bm:.6f} N.m.s/rad")
    print(f"   - B2 refletido (n_12^2 * B2): {B2_refletido:.8f} N.m.s/rad")
    print(f"   - B4 refletido (n_tot^2 * B4):{B4_refletido:.8f} N.m.s/rad")
    print(f"   >> Beq Calculado:            {Beq:.6f} N.m.s/rad (≈ 0.2527)")
    print(f"   >> Beq Nominal do PDF:       0.2530 N.m.s/rad")
    print(f"   >> Veredito: Conforme autorização do Prof. Cláudio Duarte, adota-se 0.2527 N.m.s/rad.")

    print("\n4. Parâmetros das Tramitâncias Realizáveis:")
    print(f"   - Bloco Elétrico:")
    print(f"     * Ganho do integrador (1 / La):   {ganho_int_el:.4f} [1/H]")
    print(f"     * Polo elétrico (Ra / La):         {polo_el:.4f} [rad/s]")
    print(f"   - Bloco Mecânico Equivalente:")
    print(f"     * Ganho do integrador (1 / Jeq):  {ganho_int_mec:.6f} [1/(kg.m^2)]")
    print(f"     * Polo mecânico (Beq / Jeq):       {polo_mec:.6f} [rad/s]")

    print("\n5. Função de Transferência do Motor em Malha Aberta Gm(s) = Omega_m(s) / Ea(s):")
    print(f"   Gm(s) = {num_Gm:.6f} / ( s^2 + {den_Gm_s1:.6f} s + {den_Gm_s0:.6f} )")
    print(f"   Polos:")
    print(f"     {polos_str}")
    
    print("\n6. Função de Transferência da Antena Gp(s) = Theta_L(s) / Ea(s):")
    print(f"   Gp(s) = n_tot / s * Gm(s)")
    print(f"   Gp(s) = {num_Gm * ntot:.6f} / [ s * (s^2 + {den_Gm_s1:.6f} s + {den_Gm_s0:.6f}) ]")
    print(f"   (Sistema de 3ª Ordem com integrador puro em s = 0)")

    print("\n" + "=" * 70)
