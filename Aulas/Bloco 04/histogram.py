import matplotlib.pyplot as plt
import numpy as np

N_points = 100000
n_bins = 20

#Gerar as distribuições
rng = np.random.default_rng()

dist = rng.standard_normal(N_points)

#Cria a imagem
fig, ax = plt.subplots()

#Coloca na imagens os dados
ax.hist(dist, bins=n_bins)
ax.set_xlabel('Número de Ocorrências')
ax.set_ylabel('Valores')
ax.set_title('Histograma')

plt.show()