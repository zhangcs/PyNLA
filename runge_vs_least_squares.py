"""Near-linear data: polynomial interpolation versus a least-squares line.

Dependencies: numpy, matplotlib
Run: python runge_vs_least_squares.py
The interactive Matplotlib window supports zooming and panning.
"""

import numpy as np


def barycentric_interpolate(x_nodes, y_nodes, x):
    """Evaluate the interpolating polynomial without a Vandermonde solve."""
    weights = np.ones_like(x_nodes, dtype=float)
    for j in range(x_nodes.size):
        weights[j] = 1.0 / np.prod(x_nodes[j] - np.delete(x_nodes, j))
    weights /= np.max(np.abs(weights))

    values = np.empty_like(x, dtype=float)
    for i, xi in enumerate(x):
        node_match = np.flatnonzero(xi == x_nodes)
        if node_match.size:
            values[i] = y_nodes[node_match[0]]
        else:
            terms = weights / (xi - x_nodes)
            values[i] = np.dot(terms, y_nodes) / np.sum(terms)
    return values


def main():
    import matplotlib.pyplot as plt

    # Adjust these two parameters to explore interpolation of noisy data.
    n_points = 15
    noise_std = 0.025
    rng = np.random.default_rng(7)  # Reproducible observations.
    x_data = np.linspace(-1.0, 1.0, n_points)
    y_data = 1.5 * x_data + 0.4 + rng.normal(0.0, noise_std, n_points)
    x_plot = np.linspace(-1.0, 1.0, 4001)
    y_true = 1.5 * x_plot + 0.4

    y_interp = barycentric_interpolate(x_data, y_data, x_plot)

    # min_{a,b} sum_i (a*x_i + b - y_i)^2; no normal equations.
    design = np.column_stack((x_data, np.ones_like(x_data)))
    slope, intercept = np.linalg.lstsq(design, y_data, rcond=None)[0]
    y_fit = slope * x_plot + intercept

    print(f"Observations: {n_points}, noise standard deviation: {noise_std}")
    print(f"Least-squares line: y = {slope:.6f} x + {intercept:.6f}")
    print("Errors relative to the underlying line on a dense grid:")
    for name, prediction in (("Interpolation", y_interp), ("Least squares", y_fit)):
        error = prediction - y_true
        print(f"  {name:14s} max = {np.max(np.abs(error)):.6e}, "
              f"RMSE = {np.sqrt(np.mean(error**2)):.6e}")
    print("Training RMSE (relative to observed data):")
    p_data = barycentric_interpolate(x_data, y_data, x_data)
    print(f"  Interpolation: {np.sqrt(np.mean((p_data - y_data)**2)):.6e}")
    print(f"  Least squares: {np.sqrt(np.mean((design @ [slope, intercept] - y_data)**2)):.6e}")

    fig, error_ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    error_ax.set_title("Near-linear noisy data: deviation from y = 1.5x + 0.4")
    error_ax.axhline(0.0, color="black", linestyle="--", linewidth=1,
                     label="Underlying line (zero deviation)")
    error_ax.plot(x_plot, y_interp - y_true, color="#d1495b",
                  label=f"Degree-{n_points - 1} interpolation error")
    error_ax.plot(x_plot, y_fit - y_true, color="#00798c", label="Least-squares error")
    error_ax.scatter(x_data, y_data - (1.5 * x_data + 0.4), color="#edae49",
                     edgecolor="black", s=30, zorder=4, label="Observation noise")
    error_ax.set_xlabel("x")
    error_ax.set_ylabel("Deviation from underlying line")
    error_ax.legend(loc="best")
    error_ax.set_xlim(-1.0, 1.0)
    error_ax.grid(True, alpha=0.25)
    plt.show()


if __name__ == "__main__":
    main()
