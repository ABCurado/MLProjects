import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
from itertools import cycle
# Color Cycle
cycol = cycle('bgkyrcm')
df = pd.read_csv('../log_files/fitness_&_operator_probs_19_1_log.csv')

df = df.drop(["elite_fitness", "generation", "population_fitness", "val_fitness"], axis=1)
x = df.index
operator_lines = []
fig, ax = plt.subplots()



for i, line in enumerate(df):
    operator_line, = ax.plot(x, df[df.columns[i]], color=next(cycol))
    operator_lines.append(operator_line)


def init():
    for line in operator_lines:
        line.set_data([],[])
    return operator_lines
for column in df.columns:
    print(df[column].describe())

plt.xlabel('Generation')
plt.ylabel('Percentage')
plt.gca().legend(set(df.columns))
def update(num, x, lines):
    for i, operator_line in enumerate(lines):
        operator_line.set_data(x[:num], df[df.columns[i]][:num])
    return lines

ani = animation.FuncAnimation(fig, update,init_func=init, fargs=[x, operator_lines],
                              interval=1, blit=True)
ani.save('operator_percentages.gif', writer='imagemagick')
plt.show()