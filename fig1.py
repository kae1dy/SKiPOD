import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import matplotlib as mpl

fig, ax = plt.subplots(figsize=(10,8))

ax.plot(threads, raw, label='Без распараллеливания')
ax.plot(threads, extra, label='C распараллеливанием', marker='o')
ax.set_yscale('log')
ax.grid()

ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x: .1f}'))
ax.set_xlabel('Кол-во потоков')
ax.set_ylabel('Время(в секундах)')

plt.title('Сравнение скорости алгоритма с распараллеливанием и без\n(EXTRALARGE_DATASET)')
plt.legend()

plt.savefig("./fig1.png")
