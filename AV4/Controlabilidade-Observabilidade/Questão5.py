
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os

# ==========================================================
# QUESTAO 5 - FILTRO NOTCH TWIN-T
# RC = 4 e K = R4/R5 = 1
# ==========================================================

# Criar pasta para salvar as figuras
os.makedirs("Figuras_Questao5", exist_ok=True)

# Matrizes do sistema
A = np.array([
    [-1/4,  1/4, -1/4],
    [ 1/4,  1/4, -1/4],
    [ 3/8,  3/8, -1/4]
])

B = np.array([
    [ 1/4],
    [-1/4],
    [-1/4]
])

C = np.array([[-2, -2, 0]])
D = 2

# Entrada u(t) = 1 + sen(2t)
def entrada(t):
    return 1 + np.sin(2*t)

# Equacoes diferenciais do sistema
def sistema(t, x):
    return (A @ x + B.flatten() * entrada(t))

# Condicao inicial
x0 = np.array([0, 0, 0])

# Intervalo de simulacao
t = np.linspace(0, 40, 2000)

# Resolver sistema
sol = solve_ivp(
    sistema,
    [t[0], t[-1]],
    x0,
    t_eval=t,
    rtol=1e-9,
    atol=1e-11
)

# Extrair estados
x1 = sol.y[0]
x2 = sol.y[1]
x3 = sol.y[2]

# Calcular entrada e saida
u = entrada(t)
y = (C @ sol.y).flatten() + D*u

# ==========================================================
# ITEM A - GRAFICO DA ENTRADA E SAIDA
# ==========================================================

plt.figure(figsize=(10, 5))

plt.plot(t, u, label='Entrada u(t)')
plt.plot(t, y, '--', label='Saida y(t)')

plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude')
plt.title('Questao 5(a) - Entrada e Saida')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig('Figuras_Questao5/figura_5a_entrada_saida.png',
            dpi=300)
plt.show()

# ==========================================================
# ITEM A - GRAFICO DOS TRES ESTADOS
# ==========================================================

plt.figure(figsize=(10, 5))

plt.plot(t, x1, label='x1(t)')
plt.plot(t, x2, label='x2(t)')
plt.plot(t, x3, label='x3(t)')

plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude dos estados')
plt.title('Questao 5(a) - Evolucao dos Estados')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig('Figuras_Questao5/figura_5a_estados.png',
            dpi=300)
plt.show()

# ==========================================================
# ITEM A - PROJECOES NOS PLANOS XY
# ==========================================================

fig, ax = plt.subplots(1, 3, figsize=(15, 4))

# Plano x1 x x2
ax[0].plot(x1, x2)
ax[0].set_xlabel('x1')
ax[0].set_ylabel('x2')
ax[0].set_title('Plano (x1, x2)')
ax[0].grid(True)

# Plano x1 x x3
ax[1].plot(x1, x3)
ax[1].set_xlabel('x1')
ax[1].set_ylabel('x3')
ax[1].set_title('Plano (x1, x3)')
ax[1].grid(True)

# Plano x2 x x3
ax[2].plot(x2, x3)
ax[2].set_xlabel('x2')
ax[2].set_ylabel('x3')
ax[2].set_title('Plano (x2, x3)')
ax[2].grid(True)

fig.suptitle('Questao 5(a) - Projecoes dos Estados')
plt.tight_layout()

plt.savefig('Figuras_Questao5/figura_5a_projecoes.png',
            dpi=300)
plt.show()

# ==========================================================
# ITEM G - TRANSFORMACAO PARA A NOVA BASE
# ==========================================================

T = np.array([
    [-1, 0, 1/3],
    [ 1, 2/3, 1/3],
    [ 1, 1, 0]
])

T_inv = np.linalg.inv(T)

# Transformacao dos estados:
# z = T_inv * x

z = T_inv @ sol.y

z1 = z[0]
z2 = z[1]
z3 = z[2]

# ==========================================================
# ITEM G - ESTADO z1 AO LONGO DO TEMPO
# ==========================================================

plt.figure(figsize=(10, 5))

plt.plot(t, z1, label='z1(t)')

plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude')
plt.title('Questao 5(g) - Estado z1 na Nova Base')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig('Figuras_Questao5/figura_5g_estado_z1.png',
            dpi=300)
plt.show()

# ==========================================================
# ITEM G - PROJECAO XY DOS ESTADOS z2 E z3
# ==========================================================

plt.figure(figsize=(7, 6))

plt.plot(z2, z3, label='Trajetoria (z2, z3)')

plt.xlabel('z2')
plt.ylabel('z3')
plt.title('Questao 5(g) - Projecao na Nova Base')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig('Figuras_Questao5/figura_5g_projecao_xy.png',
            dpi=300)
plt.show()

# ==========================================================
# VERIFICACAO DOS RESULTADOS
# ==========================================================

print("\n===== RESULTADOS DA QUESTAO 5 =====")

print("\nEntrada e saida:")
print("Ganho esperado: 2")

print("\nPosto da matriz de controlabilidade:")
Co = np.hstack([B, A @ B, A @ A @ B])
print(np.linalg.matrix_rank(Co))

print("\nPosto da matriz de observabilidade:")
Ob = np.vstack([C, C @ A, C @ A @ A])
print(np.linalg.matrix_rank(Ob))

print("\nAutovalores:")
print(np.linalg.eigvals(A))

print("\nVerificacao da nova base:")
print("Maximo |z2|:", np.max(np.abs(z2)))
print("Maximo |z3|:", np.max(np.abs(z3)))

print("\nTodas as figuras foram salvas na pasta:")
print("Figuras_Questao5")
