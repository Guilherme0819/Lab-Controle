import numpy as np
import matplotlib.pyplot as plt
import control as ct
import scipy.linalg as la

# ==========================================
# 1. PARÂMETROS DO CIRCUITO (Questão 4)
# ==========================================
RC = 4.0
K = 0.75  # K = R4/R5 = 3/4

# ==========================================
# 2. MATRIZES DE ESTADOS (CORRIGIDAS)
# ==========================================
A = np.array([
    [ -(1/RC)*(2 - K),   (1/RC)*K,        -(1/RC) ],
    [  (1/RC)*K,        -(1/RC)*(2 - K),  -(1/RC) ],
    [  (1/(2*RC))*(1 + 2*K), (1/(2*RC))*(1 + 2*K), -(1/RC) ]
])

B = np.array([
    [  (1/RC)*(2 - K) ],
    [ -(1/RC)*K ],
    [  (1/(2*RC))*K ] 
])

C = np.array([ [-(1 + K), -(1 + K), 0] ])
D = np.array([ [1 + K] ])

sys = ct.ss(A, B, C, D)

# ==========================================
# 3. OBSERVABILIDADE E KALMAN (No Terminal)
# ==========================================
O_mat = ct.obsv(A, C)
rank_O = np.linalg.matrix_rank(O_mat)

print("="*50)
print(" ANÁLISE DE OBSERVABILIDADE - QUESTÃO 4")
print("="*50)
print(f"Posto: {rank_O} -> Inobservável!")

# ==========================================
# 4a. SIMULAÇÃO: DEGRAU UNITÁRIO
# ==========================================
t_deg = np.linspace(0, 50, 1000)
t_deg, y_deg, x_deg = ct.step_response(sys, T=t_deg, return_x=True)

x1_deg = x_deg[0, :]
x2_deg = x_deg[1, :]
x3_deg = x_deg[2, :]

# GRÁFICO 1: Saída ao Degrau
plt.figure(figsize=(10, 5))
plt.plot(t_deg, np.squeeze(y_deg), 'b-', linewidth=2, label='Saída $y(t)$')
plt.title("Questão 4a - Resposta do Filtro Notch ao Degrau Unitário")
plt.xlabel("Tempo (s)")
plt.ylabel("Tensão (V)")
plt.legend()
plt.grid(True)
plt.show()

# GRÁFICO 2: Estados ao Degrau
plt.figure(figsize=(10, 5))
plt.plot(t_deg, x1_deg, 'r', label='$x_1$ ($V_{C1}$)')
plt.plot(t_deg, x2_deg, 'g', label='$x_2$ ($V_{C2}$)')
plt.plot(t_deg, x3_deg, 'm', label='$x_3$ ($V_{C3}$)')
plt.title("Questão 4a - Estados na Resposta ao Degrau")
plt.xlabel("Tempo (s)")
plt.ylabel("Tensão (V)")
plt.legend()
plt.grid(True)
plt.show()

# GRÁFICO 3: Projeções 2D
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
ax1.plot(x1_deg, x2_deg, 'b')
ax1.set_title("Projeção $(x_1, x_2)$")
ax1.grid(True)
ax2.plot(x1_deg, x3_deg, 'g')
ax2.set_title("Projeção $(x_1, x_3)$")
ax2.grid(True)
ax3.plot(x2_deg, x3_deg, 'r')
ax3.set_title("Projeção $(x_2, x_3)$")
ax3.grid(True)
plt.tight_layout()
plt.show()

# ==========================================
# 4b. SIMULAÇÃO: SENOIDE
# ==========================================
t_sen = np.linspace(0, 20, 1000)
u_sen = 1 + np.sin(2 * t_sen)
t_sen, y_sen, x_sen = ct.forced_response(sys, T=t_sen, U=u_sen, X0=[0, 0, 0], return_x=True)

# GRÁFICO 4: Saída à Senoide
plt.figure(figsize=(10, 5))
plt.plot(t_sen, u_sen, 'k--', label=r'Entrada $u(t) = 1 + \sin(2t)$')
plt.plot(t_sen, np.squeeze(y_sen), 'b-', linewidth=2, label='Saída $y(t)$')
plt.title("Questão 4b - Resposta à entrada senoidal")
plt.xlabel("Tempo (s)")
plt.ylabel("Tensão (V)")
plt.legend()
plt.grid(True)
plt.show()

# ==========================================
# 4g. SIMULAÇÃO: INOBSERVABILIDADE
# ==========================================
t_sim = np.linspace(0, 30, 1000)
t1, y1, x1_sim = ct.initial_response(sys, T=t_sim, X0=[10, -8, 15], return_x=True)
t2, y2, x2_sim = ct.initial_response(sys, T=t_sim, X0=[-10, 10, 0], return_x=True)

# GRÁFICO 5: Condições Iniciais
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.plot(t1, np.squeeze(y1), 'b-', linewidth=2, label='Saída $y(t)$')
ax1.set_title("Questão 4g: $x(0) = [10, -8, 15]^T$")
ax1.grid(True)
ax2.plot(t2, np.squeeze(y2), 'r-', linewidth=2, label='Saída $y(t)$')
ax2.set_title("Questão 4g: $x(0) = [-10, 10, 0]^T$ (Modo Inobservável!)")
ax2.grid(True)
plt.tight_layout()
plt.show()