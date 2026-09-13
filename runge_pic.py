import numpy as np
import matplotlib.pyplot as plt

# Runge 函数
def runge(x):
    return 1.0 / (1 + 25 * x**2)

# Lagrange 插值
def lagrange_interp(x_nodes, y_nodes, x_eval):
    n = len(x_nodes)
    p = np.zeros_like(x_eval)
    for i in range(n):
        xi, yi = x_nodes[i], y_nodes[i]
        Li = np.ones_like(x_eval)
        for j in range(n):
            if i != j:
                xj = x_nodes[j]
                Li *= (x_eval - xj) / (xi - xj)
        p += yi * Li
    return p

# 等距节点设置
node_counts = [3, 7, 11]
colors = ["orange", "green", "red"]
labels = [
    "3 nodes (degree 2)",
    "7 nodes (degree 6)",
    "11 nodes (degree 10)",
]

x_dense = np.linspace(-1, 1, 2000)
y_true = runge(x_dense)

plt.figure(figsize=(9, 6), dpi=150)

# 真实函数
plt.plot(x_dense, y_true, "k-", label="True Runge function $f(x)=1/(1+25x^2)$", linewidth=1.4)

# 不同节点数插值
for n, color, label in zip(node_counts, colors, labels):
    x_nodes = np.linspace(-1, 1, n)
    y_nodes = runge(x_nodes)
    y_interp = lagrange_interp(x_nodes, y_nodes, x_dense)

    plt.plot(x_dense, y_interp, linestyle="--", color=color, label=label, linewidth=1.1)
    plt.plot(x_nodes, y_nodes, "o", color=color, markersize=5, zorder=6)

plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title("Runge Phenomenon: Equidistant Interpolation with 3, 7, 11 Nodes")

plt.legend(fontsize=8, loc="upper center")
plt.grid(alpha=0.35, linewidth=0.4)

plt.xlim(-1.02, 1.02)
plt.ylim(-0.5, 2.0)   # 扩大 y 范围，保证曲线完整显示

plt.tight_layout()
plt.show()

