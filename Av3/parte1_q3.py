import numpy as np
import matplotlib.pyplot as plt
import control as ct
from scipy import signal

# ==========================================
# 1. BASE DA PLANTA (Copiado da Q2 para independência)
# ==========================================
Ra = 10.0; La = 0.125; Kb = 0.45; Kt_motor = 15.0
Jm = 2.5; J2 = 0.075; J4 = 20.0
Bm = 0.25; B2 = 0.085; B4 = 0.35

n12 = 15/90
ntot = (15/90) * (30/160)

Jeq = Jm + (n12**2)*J2 + (ntot**2)*J4
Beq = Bm + (n12**2)*B2 + (ntot**2)*B4

s = ct.tf('s')
G_el = 1 / (La*s + Ra)
G_mec = 1 / (Jeq*s + Beq)

G_direta = ct.series(G_el, Kt_motor, G_mec)
Gm = ct.feedback(G_direta, Kb, sign=-1) # Função de Transferência do Motor

# ==========================================
# 2. PARÂMETROS DO CONTROLADOR (Questão 3)
# ==========================================
Kp = 0.05        # Sensor de posição [V/rad]
Kt_taco = 0.04   # Tacogerador [V/rad/s]
K4 = 30.0        # Ganho do Amplificador de Potência 
E_ref = 60.0     # Set-point (Referência)

# Casos a simular com os diferentes ganhos
casos = [
    {'letra': 'A', 'K1': 4.6062,  'K2': 0.2257},
    {'letra': 'B', 'K1': 29.4795, 'K2': 0.2807},
    {'letra': 'C', 'K1': 4.6062,  'K2': 0.0528},
    {'letra': 'D', 'K1': 57.5772, 'K2': 0.1683},
    {'letra': 'E', 'K1': 2.8789,  'K2': 0.5750}
]

T = np.linspace(0, 10, 2000) # Simulação de 0 a 10 segundos

for caso in casos:
    K1 = caso['K1']
    K2 = caso['K2']
    K5 = Kp * K1 * (np.pi / 180.0)
    
    # ==========================================
    # 3. FUNÇÃO DE TRANSFERÊNCIA GLOBAL M(s)
    # ==========================================
    # Numerador (Caminho Direto)
    Numerador = K5 * K4 * Gm * (ntot / s)
    
    # Denominador = 1 - (Laços) -> Ambos são realimentações negativas
    Laco_Velocidade = K4 * Gm * K2 * Kt_taco
    Laco_Posicao = K4 * Gm * (ntot / s) * K1 * Kp
    Denominador = 1 + Laco_Velocidade + Laco_Posicao
    
    # Função M(s) = Theta_L(s) / R(s). ct.minreal cancela polos/zeros repetidos
    M_pos = ct.minreal(Numerador / Denominador)
    
    # Para achar a Velocidade, derivamos a posição (multiplicando por s)
    # e invertemos a engrenagem (dividindo por ntot)
    M_vel = ct.minreal(M_pos * (s / ntot))
    
    # ==========================================
    # 4. POLOS, ZEROS E SIMULAÇÃO
    # ==========================================
    polos = M_pos.poles()
    zeros = M_pos.zeros()
    
    print(f"\n{'='*40}")
    print(f" CASO {caso['letra']}: K1 = {K1}, K2 = {K2}")
    print(f"{'='*40}")
    print("Polos em Malha Fechada:")
    for p in polos:
        print(f"  {p:.4f}")
    
    print("\nZeros em Malha Fechada:")
    if len(zeros) == 0:
        print("  Não possui zeros.")
    else:
        for z in zeros:
            print(f"  {z:.4f}")
            
    # Simulação do Degrau (r(t) = 60)
    t, theta_L_rad = ct.step_response(E_ref * M_pos, T)
    t, omega_m_rad_s = ct.step_response(E_ref * M_vel, T)
    
    # Conversões
    theta_L_deg = theta_L_rad * (180.0 / np.pi) 
    omega_m_rpm = omega_m_rad_s * (60.0 / (2 * np.pi)) 
    
    # ==========================================
    # 5. GRÁFICOS
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(t, theta_L_deg, 'b', linewidth=2)
    ax1.axhline(y=E_ref, color='r', linestyle='--', label=f'Set-point ({E_ref}°)')
    ax1.set_title(f"Caso {caso['letra']} - Posição da Antena")
    ax1.set_ylabel("Posição (Graus)")
    ax1.grid(True)
    ax1.legend(loc='lower right')
    
    ax2.plot(t, omega_m_rpm, 'g', linewidth=2)
    ax2.set_title(f"Caso {caso['letra']} - Velocidade do Motor")
    ax2.set_xlabel("Tempo (s)")
    ax2.set_ylabel("Velocidade (RPM)")
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show() # Ao fechar a janela deste gráfico, o próximo caso é simulado automaticamente