import numpy as np
import matplotlib.pyplot as plt
import control as ct

# ==========================================
# 1. MATRIZES DA PLANTA (Espaço de Estados)
# Matriz A corrigida com a dedução física exata
# ==========================================
A = np.array([[-80.0,    -3.6,     0.0],
              [ 5.9197,  -0.0997,  0.0],
              [ 0.0,      0.04,    0.0]])

B = np.array([[8.0],
              [0.0],
              [0.0]])

C = np.array([[0.0, 0.0, 1.0]]) # Saída é apenas o estado 3 (Posição)
D = np.array([[0.0]])

# ==========================================
# 2. PARÂMETROS DO CONTROLADOR
# ==========================================
Ki = 0.025   # Constante do sensor de corrente
Kt = 0.04    # Constante do Tacogerador
Kp = 0.05    # Constante do Sensor de posição
K4 = 20.0    # Ganho do Amplificador de Potência

# O enunciado pede uma entrada degrau r(t) = (pi/3) 
# que equivale exatamente a 60 graus!
r_t = np.pi / 3.0  

# ==========================================
# 3. CASOS DE SIMULAÇÃO (Malha Fechada)
# ==========================================
casos = [
    {'letra': 'A', 'K1': -16.9000, 'K2':  0.3949, 'K3':  8.3296},
    {'letra': 'B', 'K1': -17.5250, 'K2': -0.0674, 'K3': 16.6637},
    {'letra': 'C', 'K1': -18.8375, 'K2': -0.0798, 'K3':  6.2455},
    {'letra': 'D', 'K1': -19.4625, 'K2': -0.1221, 'K3':  6.2453},
    {'letra': 'E', 'K1': -20.2125, 'K2': -0.1729, 'K3':  6.2451}
]

T = np.linspace(0, 10, 2000) # Simulação de 0 a 10 segundos

for caso in casos:
    K1 = caso['K1']
    K2 = caso['K2']
    K3 = caso['K3']
    
    # Condição do PDF: Ajustar K5 de modo que K5 = K3 * Kp
    K5 = K3 * Kp
    
    # ----------------------------------------------------
    # Construção da Malha Fechada em Espaço de Estados
    # ----------------------------------------------------
    # O Vetor de ganhos de realimentação K (1x3)
    K_vec = K4 * np.array([[K1 * Ki, K2 * Kt, K3 * Kp]])
    
    # Matriz A em malha fechada: A_cl = A - B*K
    A_cl = A - (B @ K_vec)
    
    # Matriz B em malha fechada: B_cl = B * K4 * K5
    B_cl = B * (K4 * K5)
    
    # Criação do Sistema
    sys_cl = ct.ss(A_cl, B_cl, C, D)
    
    # ----------------------------------------------------
    # Extração de Polos
    # ----------------------------------------------------
    polos = sys_cl.poles()
    
    print(f"\n{'='*40}")
    print(f" CASO {caso['letra']}: K1={K1}, K2={K2}, K3={K3}")
    print(f"{'='*40}")
    print("Polos em Malha Fechada:")
    for p in polos:
        print(f"  {p:.4f}")
        
    # ----------------------------------------------------
    # Simulação (Extraindo todos os estados)
    # return_x=True faz a função devolver o vetor de estados x
    # ----------------------------------------------------
    t, y, x = ct.step_response(sys_cl * r_t, T, return_x=True)
    
    # x tem 3 linhas (os 3 estados). Vamos separá-los:
    i_a = x[0, :]      # Estado 1: Corrente
    omega_m = x[1, :]  # Estado 2: Velocidade
    theta_L = x[2, :]  # Estado 3: Posição (que também é a saída y)
    
    # Conversões de unidades para visualização intuitiva
    theta_L_deg = theta_L * (180.0 / np.pi) 
    omega_m_rpm = omega_m * (60.0 / (2 * np.pi))
    
    # ----------------------------------------------------
    # Geração dos Gráficos
    # ----------------------------------------------------
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
    
    # Gráfico 1: Posição
    ax1.plot(t, theta_L_deg, 'b', linewidth=2)
    ax1.axhline(y=60.0, color='r', linestyle='--', label='Set-point (60°)')
    ax1.set_title(f"Caso {caso['letra']} - Posição da Antena ($\\theta_L$)")
    ax1.set_ylabel("Posição (Graus)")
    ax1.grid(True)
    ax1.legend(loc='lower right')
    
    # Gráfico 2: Velocidade
    ax2.plot(t, omega_m_rpm, 'g', linewidth=2)
    ax2.set_title(f"Caso {caso['letra']} - Velocidade do Motor ($\\omega_m$)")
    ax2.set_ylabel("Velocidade (RPM)")
    ax2.grid(True)
    
    # Gráfico 3: Corrente (Novo na Parte 2)
    ax3.plot(t, i_a, 'm', linewidth=2)
    ax3.set_title(f"Caso {caso['letra']} - Corrente de Armadura ($i_a$)")
    ax3.set_ylabel("Corrente (A)")
    ax3.set_xlabel("Tempo (s)")
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()