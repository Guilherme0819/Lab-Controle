import numpy as np
import matplotlib.pyplot as plt
import control as ct
from scipy import signal

# Parâmetros
Ra = 10.0; La = 0.125; Kb = 0.45; Kt = 15.0
Jm = 2.5; J2 = 0.075; J4 = 20.0
Bm = 0.25; B2 = 0.085; B4 = 0.35

# Relações de transmissão
n12 = 15/90
ntot = (15/90) * (30/160)

# Parâmetros Equivalentes
Jeq = Jm + (n12**2)*J2 + (ntot**2)*J4
Beq = Bm + (n12**2)*B2 + (ntot**2)*B4

# Subsistemas e Função de Transferência
s = ct.tf('s')
G_el = 1 / (La*s + Ra)
G_mec = 1 / (Jeq*s + Beq)

G_direta = ct.series(G_el, Kt, G_mec)
Gm = ct.feedback(G_direta, Kb, sign=-1)

# Simulação do Degrau
T = np.linspace(0, 15, 1000)
t, omega_m = ct.step_response(110 * Gm, T)
omega_m_rpm = omega_m * (60 / (2 * np.pi))

# Plot
plt.plot(t, omega_m_rpm)
plt.title('Velocidade em Malha Aberta (110V)')
plt.xlabel('Tempo (s)')
plt.ylabel('Velocidade (RPM)')
plt.grid(True)
plt.show()

# Frações Parciais (residue)
num_step = np.polymul(Gm.num[0][0], [110])
den_step = np.polymul(Gm.den[0][0], [1, 0])
r, p, k = signal.residue(num_step, den_step)

print("Resíduos:", r)
print("Polos:", p)