import numpy as np
import matplotlib.pyplot as plt

# ==============================================
# 1. 生成中心化二维数据，并剔除y最大的那个点
# ==============================================
np.random.seed(42)
n_samples = 80
mean = [0, 0]
cov = [[3, 2.6],
       [2.6, 3]]
X = np.random.multivariate_normal(mean, cov, n_samples)

# 移除x2（y坐标）最大的离群点
idx_remove = np.argmax(X[:, 1])
X = np.delete(X, idx_remove, axis=0)

# ==============================================
# 2. 初等方法求第一主成分（不调用sklearn PCA）
# 协方差矩阵 S = (1/n) X^T X，求特征分解
# ==============================================
n = X.shape[0]
S = (X.T @ X) / n
eigvals, eigvecs = np.linalg.eig(S)
idx_max = np.argmax(eigvals)
v1 = eigvecs[:, idx_max]   # 第一主成分单位方向向量

# ==============================================
# 3. 正交投影：点 p 投影到过原点、方向v1的直线上
# ==============================================
def project_point(p, dir_vec):
    t = np.dot(p, dir_vec)
    foot = t * dir_vec
    return foot

feet = np.array([project_point(p, v1) for p in X])

# ==============================================
# 4. 绘图
# ==============================================
plt.figure(figsize=(6,6), dpi=120)
ax = plt.gca()

# 蓝色散点
ax.scatter(X[:,0], X[:,1], color='#4488cc', s=25, alpha=0.8)

# 红色主成分直线：拉长
scale = 6.0
ax.plot([-scale*v1[0], scale*v1[0]],
        [-scale*v1[1], scale*v1[1]],
        color='#dd3333', lw=2)

# 绿色短线：严格正交投影
for p, f in zip(X, feet):
    ax.plot([p[0], f[0]], [p[1], f[1]], color='#22aa44', lw=1.2)

ax.scatter(0,0, c='red', s=40, zorder=5) # 原点红点

ax.set_xlim(-6,6)
ax.set_ylim(-6,6)
ax.set_xlabel("$x_1$", fontsize=11)
ax.set_ylabel("$x_2$", fontsize=11)
ax.grid(alpha=0.3)
ax.set_aspect("equal") # 等比例坐标轴，保证视觉垂直

plt.tight_layout()
plt.show()
