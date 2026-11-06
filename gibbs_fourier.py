import numpy as np
import matplotlib.pyplot as plt


# 类似 Runge 函数的单峰形状，但周期延拓仍解析光滑。
def smooth_function(x):
    x = np.asarray(x, dtype=float)
    return 1.0 / (1 + 25 * np.sin(np.pi * x / 2)**2)


# 由 Poisson 核展开得 f(x) = (1 + 2 sum r^k cos(k*pi*x))/sqrt(26)。
# 系数几何衰减，截断误差的精确上界在 x=0 处达到。
def smooth_fourier_sum(x, n):
    x = np.asarray(x, dtype=float)
    r = (np.sqrt(26.0) - 1) / (np.sqrt(26.0) + 1)
    p = np.ones_like(x)
    for k in range(1, n + 1):
        p += 2 * r**k * np.cos(k * np.pi * x)
    return p / np.sqrt(26.0)


# 周期为 2 的方波：(-1, 0) 上为 -1，(0, 1) 上为 1。
# 每个整数都是间断点，取左右极限的平均值 0。
def square_wave(x):
    x = np.asarray(x, dtype=float)
    phase = np.remainder(x, 2.0)
    return np.where(phase == np.floor(phase), 0.0,
                    np.where(phase < 1.0, 1.0, -1.0))


# n 是最高 Fourier 频率；偶数频率的系数为 0。
def fourier_sum(x, n):
    x = np.asarray(x, dtype=float)
    p = np.zeros_like(x)
    for k in range(1, n + 1, 2):
        p += np.sin(k * np.pi * x) / k
    return (4.0 / np.pi) * p


def main():
    orders = [5, 21, 81]
    colors = ["orange", "green", "red"]
    x_dense = np.linspace(-1, 1, 20001)

    print("Smooth periodic function: maximum error on a 20001-point grid")
    print(f"{'N':>5} {'Max error':>14}")
    for n in orders:
        error = np.max(np.abs(smooth_fourier_sum(x_dense, n) - smooth_function(x_dense)))
        print(f"{n:5d} {error:14.6e}")

    for smooth in [True, False]:
        fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
        if smooth:
            ax.plot(x_dense, smooth_function(x_dense), "k-", linewidth=1.4,
                    label="Smooth periodic function")
            ax.set_title(r"Smooth: $f(x)=1/[1+25\sin^2(\pi x/2)]$")
            ax.set_ylim(-0.15, 1.4)
            partial_sum = smooth_fourier_sum
        else:
            # 分开画两段，避免用实线连接间断点。
            ax.plot([-1, 0], [-1, -1], "k-", linewidth=1.4,
                    label="Square wave (period 2)")
            ax.plot([0, 1], [1, 1], "k-", linewidth=1.4)
            ax.plot([-1, 0, 0, 1], [-1, -1, 1, 1], "o",
                    color="black", markerfacecolor="white", markersize=4)
            ax.plot([-1, 0, 1], square_wave([-1, 0, 1]), "ko", markersize=4, zorder=6)
            ax.set_title("Discontinuous: square wave (Gibbs phenomenon)")
            ax.set_ylim(-1.4, 1.65)
            partial_sum = fourier_sum

        for n, color in zip(orders, colors):
            y = partial_sum(x_dense, n)
            ax.plot(x_dense, y, "--", color=color, linewidth=1.1,
                    label=f"N = {n}")

        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_xlim(-1.02, 1.02)
        ax.grid(alpha=0.35, linewidth=0.4)
        ax.legend(fontsize=8, loc="upper left")
        fig.suptitle("Fourier expansion approximation")
        fig.tight_layout()

    # 对奇数 n，第一个正峰的位置恰为 1/(n+1)，无须网格搜索。
    # 过冲比例以跳跃幅度 2 归一化，而不是以平台高度 1 归一化。
    print("First positive peak of the Fourier partial sum (jump size = 2)")
    print(f"{'N':>5} {'Peak x':>12} {'Peak value':>14} {'Overshoot / jump':>18}")
    for n in orders:
        peak_x = 1.0 / (n + 1)
        peak = float(fourier_sum(peak_x, n))
        print(f"{n:5d} {peak_x:12.6f} {peak:14.8f} {(peak - 1) / 2:18.6%}")
    print("Limiting overshoot / jump: 8.948987%")
    print("The oscillatory region narrows, but the peak overshoot persists.")
    plt.show()


if __name__ == "__main__":
    main()
