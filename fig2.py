
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from matplotlib.colors import LogNorm, Normalize

import seaborn as sns
import matplotlib.pylab as plt

Z = np.array([mini, small, medium, large, extra])

fig, ax = plt.subplots(figsize=(15,10))
ax = sns.heatmap(Z,
                 linewidth=0.5,
                 annot=True, fmt='.3f',
                 xticklabels=threads,
                 yticklabels=nums,
                 cbar_kws={'label': 'Время(в секундах)'},
                 norm=LogNorm(), cmap='winter', ax=ax
                 )
ax.set_xlabel('Кол-во потоков')
ax.set_ylabel('Кол-во чисел в матрицах(всего)')
ax.invert_yaxis()
plt.title('Зависимость скорости выполнения от\nкол-ва потоков/объёма входных данных')

plt.savefig("./fig2.png")