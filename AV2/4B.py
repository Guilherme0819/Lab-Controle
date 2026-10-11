import numpy as np
import matplotlib.pyplot as plt
import control as ct

J = 0.25
B = 0.05
K = 5.0
# Em circuito aberto, não há excitação elétrica, simulamos uma perturbação
num = [1] 
den = [J, B, K]
sys_aberto = ct.tf(num, den)

t, y = ct.step_response(sys_aberto, T=np.linspace(0, 10, 1000))

plt.style.use('bmh')
plt.figure(figsize=(10, 5))
plt.plot(t, y, label=r'Oscilação Livre - Circuito Aberto ($\zeta \approx 0.02$)')
plt.title('Ensaio de Período Próprio do Galvanômetro')
plt.xlabel('Tempo (s)')
plt.ylabel('Deflexão Angular')
plt.grid(True)
plt.legend()
plt.show()