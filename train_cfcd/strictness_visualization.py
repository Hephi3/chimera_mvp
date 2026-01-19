import matplotlib.pyplot as plt
import os
import numpy as np

mean = 0
std = 1
# s = 1.0
z = lambda x: (x-mean)/std
def n(x): return 1/(std * np.sqrt(2 * np.pi)) * np.exp(-0.5 * z(x)**2)
def w(x,s): return np.exp(-s*z(x)**2)

x = np.linspace(-2, 2, 1000)
y = [n(xi) for xi in x]

yw0 = [w(xi, 0) for xi in x]
yw05 = [w(xi, 0.5) for xi in x]
yw1 = [w(xi, 1.0) for xi in x]
yw3 = [w(xi, 3.0) for xi in x]
yw10 = [w(xi, 10.0) for xi in x]

plt.figure(figsize=(6, 3))
plt.plot(x,y, label='Data Distribution', linestyle='--', color='black')
plt.plot(x,yw0, label='Weights s=0.0')
plt.plot(x,yw05, label='Weights s=0.5')
plt.plot(x,yw1, label='Weights s=1.0')
plt.plot(x,yw3, label='Weights s=3.0')
plt.plot(x,yw10, label='Weights s=10.0')
plt.xlabel('z-score of Training Sample')
plt.ylabel('Training Sample Loss Weight')
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
output_path = 'strictness_normalization_function.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {output_path}")
plt.show()
