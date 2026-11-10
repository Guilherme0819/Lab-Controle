import numpy as np
import matplotlib.pyplot as plt
import control as ct

# ==========================================
# PARÂMETROS DO CIRCUITO (Questão 3)
# ==========================================
RC = 0.25
K = 1.0  # K = R4/R5

# ==========================================
# DEFINIÇÃO DAS MATRIZES DE ESPAÇO DE ESTADOS
# ==========================================
# Baseado nas equações do PDF:
A = np.array([
    [ -(1/RC)*(2 - K),   (1/RC)*K,        -(1/RC) ],
    [  (1/RC)*K,        -(1/RC)*K,        -(1/RC) ],
    [  (1/(2*RC))*(1 + 2*K), (1/(2*RC))*(1 + 2*K), -(1/RC) ]
])

# Assumindo sinais extraídos do roteiro:
B = np.array([
    [  (1/RC)*(2 - K) ],
    [ -(1/RC)*K ],
    [  (1/(2*RC))*K ] 
])

C = np.array([ [-(1 + K), -(1 + K), 0] ])
D = np.array([ [1 + K] ])

# Criação do sistema em Espaço de Estados
sys = ct.ss(A, B, C, D)

# ==========================================
# ANÁLISE DE CONTROLABILIDADE E FUNÇÃO DE TRANSFERÊNCIA
# ==========================================
# Matriz de Controlabilidade (C = [B, AB, A^2B])
C_mat = ct.ctrb(A, B)
rank_C = np.linalg.matrix_rank(C_mat)

# Função de Transferência
sys_tf = ct.ss2tf(sys)
sys_tf_min = ct.minreal(sys_tf) # Cancelamento de polos/zeros não controláveis/observáveis

print("="*50)
print(" ANÁLISE DE CONTROLABILIDADE - QUESTÃO 3")
print("="*50)
print(f"Posto (Rank) da Matriz de Controlabilidade: {rank_C}")
if rank_C < A.shape[0]:
    print("-> CONCLUSÃO: O sistema é INCONTROLÁVEL! (Rank < 3)")
else:
    print("-> CONCLUSÃO: O sistema é Controlável. (Rank = 3)")

print("\nFunção de Transferência (Sem Cancelamentos):")
print(sys_tf)
print("\nFunção de Transferência Mínima (Polos e Zeros Cancelados):")
print(sys_tf_min)

# ==========================================
# 4. SIMULAÇÃO: u(t) = 1 + sin(2t)
# ==========================================
t = np.linspace(0, 10, 1000)
u_t = 1 + np.sin(2 * t)

# Simulação com condições iniciais nulas x(0) = [0, 0, 0]
t, y_t, x_t = ct.forced_response(sys, T=t, U=u_t, X0=[0, 0, 0], return_x=True)

# Extração dos estados
x1 = x_t[0, :]
x2 = x_t[1, :]
x3 = x_t[2, :]

# ==========================================
# GERAÇÃO DE GRÁFICOS
# ==========================================

# GRÁFICO 1: Entrada u(t) e Saída y(t)
plt.figure(figsize=(10, 5))
plt.plot(t, u_t, 'k--', label=r'Entrada $u(t) = 1 + \sin(2t)$')
plt.plot(t, np.squeeze(y_t), 'b-', linewidth=2, label=r'Saída $y(t)$')
plt.title("Questão 3a - Resposta do Filtro Notch à entrada $u(t)$")
plt.xlabel("Tempo (s)")
plt.ylabel("Tensão (V)")
plt.legend()
plt.grid(True)
plt.show()

# GRÁFICO 2: Estados x1, x2, x3
plt.figure(figsize=(10, 5))
plt.plot(t, x1, 'r', label='$x_1$ ($V_{C1}$)')
plt.plot(t, x2, 'g', label='$x_2$ ($V_{C2}$)')
plt.plot(t, x3, 'm', label='$x_3$ ($V_{C3}$)')
plt.title("Questão 3a - Evolução das Variáveis de Estado")
plt.xlabel("Tempo (s)")
plt.ylabel("Tensão (V)")
plt.legend()
plt.grid(True)
plt.show()

# GRÁFICO 3: Projeções 2D (Plano de Fase)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

ax1.plot(x1, x2, 'b')
ax1.set_title("Projeção $(x_1, x_2)$")
ax1.set_xlabel("$x_1$")
ax1.set_ylabel("$x_2$")
ax1.grid(True)

ax2.plot(x1, x3, 'g')
ax2.set_title("Projeção $(x_1, x_3)$")
ax2.set_xlabel("$x_1$")
ax2.set_ylabel("$x_3$")
ax2.grid(True)

ax3.plot(x2, x3, 'r')
ax3.set_title("Projeção $(x_2, x_3)$")
ax3.set_xlabel("$x_2$")
ax3.set_ylabel("$x_3$")
ax3.grid(True)

plt.tight_layout()
plt.show()

# GRÁFICO 4: Trajetória 3D dos Estados (Pedido no item 3g)
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x1, x2, x3, 'b', linewidth=1.5)
ax.set_title("Trajetória do Estado em 3D")
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$x_3$')
plt.show()