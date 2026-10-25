import numpy as np
import matplotlib.pyplot as plt


# Runge 函数
def runge(x):
    return 1.0 / (1 + 25 * x**2)


# 保留原程序的 Lagrange 插值公式，便于比较布点方式的影响。
def lagrange_interp(x_nodes, y_nodes, x_eval):
    n = len(x_nodes)
    p = np.zeros_like(x_eval, dtype=float)
    for i in range(n):
        xi, yi = x_nodes[i], y_nodes[i]
        Li = np.ones_like(x_eval, dtype=float)
        for j in range(n):
            if i != j:
                xj = x_nodes[j]
                Li *= (x_eval - xj) / (xi - xj)
        p += yi * Li
    return p


# n 个 Chebyshev–Lobatto 节点（n >= 2），包含端点，在两端加密。
def chebyshev_nodes(n):
    return -np.cos(np.pi * np.arange(n) / (n - 1))


def main():
    node_counts = [3, 7, 11]
    colors = ["orange", "green", "red"]
    x_dense = np.linspace(-1, 1, 20001)
    y_true = runge(x_dense)

    methods = [
        ("Equidistant nodes", lambda n: np.linspace(-1, 1, n), False),
        ("Chebyshev-Lobatto nodes", chebyshev_nodes, False),
        ("Piecewise linear (equidistant nodes)", lambda n: np.linspace(-1, 1, n), True),
    ]

    # 网格最大误差是连续区间最大误差的数值近似。
    print("Maximum absolute error on a 20001-point grid in [-1, 1]")
    print(f"{'Method / node distribution':<36} {'Nodes':>5} {'Degree':>6} {'Max error':>14}")
    for title, make_nodes, piecewise_linear in methods:
        fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
        ax.plot(x_dense, y_true, "k-", linewidth=1.4, label="Runge function")
        for n, color in zip(node_counts, colors):
            x_nodes = make_nodes(n)
            y_nodes = runge(x_nodes)
            if piecewise_linear:
                y_interp = np.interp(x_dense, x_nodes, y_nodes)
                degree = 1  # 每个子区间上的多项式次数。
                label = f"{n} nodes (piecewise linear)"
            else:
                y_interp = lagrange_interp(x_nodes, y_nodes, x_dense)
                degree = n - 1
                label = f"{n} nodes (degree {degree})"
            ax.plot(x_dense, y_interp, "--", color=color, linewidth=1.1,
                    label=label)
            ax.plot(x_nodes, y_nodes, "o", color=color, markersize=5, zorder=6)
            error = np.max(np.abs(y_interp - y_true))
            print(f"{title:<36} {n:5d} {degree:6d} {error:14.6e}")

        ax.set_title(title)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_xlim(-1.02, 1.02)
        ax.set_ylim(-0.5, 2.0)
        ax.grid(alpha=0.35, linewidth=0.4)
        ax.legend(fontsize=8, loc="upper center")

        fig.suptitle(r"Runge interpolation: $f(x)=1/(1+25x^2)$")
        fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
