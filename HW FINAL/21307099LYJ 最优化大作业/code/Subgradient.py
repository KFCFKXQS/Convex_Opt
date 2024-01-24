import numpy as np
import matplotlib.pyplot as plt
from data_generator import generate_data

# 次梯度算法实现


class SubgradientMethod:
    def __init__(self, lamda, alpha, x_dim, b_dim, stop=1e-5, max_iter=80000):
        self.lamda = lamda
        self.alpha = alpha
        self.stop = stop
        self.max_iter = max_iter
        self.x_dim = x_dim
        self.b_dim = b_dim

    def check_convergence(self, x_k, x_k_ud):
        return np.linalg.norm((x_k_ud - x_k), 2) < self.stop

    def fit(self, A, b, x_true):
        X_temp = np.zeros((self.x_dim, 1))
        count = 0
        X_opt_step = np.zeros(
            (self.max_iter, x_true.shape[0], x_true.shape[1]))

        while count < self.max_iter:
            g = np.zeros((self.x_dim, 1))
            for i in range(len(A)):  # 遍历A中的每个矩阵A_i
                g += A[i].T @ (A[i] @ X_temp - b[i].reshape(-1, 1))

            X_new = np.array([np.random.uniform(-1, 1) if x ==
                             0 else np.sign(x) for x in X_temp]).reshape(-1, 1)
            g += self.lamda * X_new
            alphak = self.alpha / np.sqrt(count + 1)
            X_update = X_temp - alphak * g
            print("Iter ", count+1, ":", np.linalg.norm((X_update - x_true), 2))
            X_opt_step[count] = X_update
            if self.check_convergence(X_update, X_temp):
                break

            X_temp = X_update
            count += 1

        return X_opt_step[:count + 1, :]


if __name__ == '__main__':
    n = 200
    m = 5
    sparsity = 0.05
    seed = 42
    alpha = 0.001

    x_true, A, e, b = generate_data(n, m, sparsity, seed=seed)
    lambda_vals = [0.001,  0.01, 0.05, 0.1, 1, 10, 20, 50]

    fig, axes = plt.subplots(2, 4, figsize=(15, 10))
    axes = axes.flatten()

    for i, lambda_val in enumerate(lambda_vals):

        pgd = SubgradientMethod(alpha=alpha, lamda=lambda_val,x_dim=n,b_dim=m)
        X_opt_step = pgd.fit(A, b, x_true)

        X_best = X_opt_step[-1]

        X_dis2best = [np.linalg.norm(X_opt - X_best) for X_opt in X_opt_step]
        X_dis2real = [np.linalg.norm(
            X_opt - x_true) for X_opt in X_opt_step]
        sparsity_levels_0 = [(np.sum(np.abs(X_opt) <= 1e-2) / len(X_best))
                             for X_opt in X_opt_step]
        sparsity_levels_1 = [(np.sum(np.abs(X_opt) <= 1e-3) / len(X_best))
                             for X_opt in X_opt_step]
        sparsity_levels_2 = [(np.sum(np.abs(X_opt) <= 1e-4) / len(X_best))
                             for X_opt in X_opt_step]
        sparsity_levels_3 = [(np.sum(np.abs(X_opt) ==0) / len(X_best))
                             for X_opt in X_opt_step]
        ax = axes[i]
        ax.plot(X_dis2best, label='Distance to X_best')
        ax.plot(X_dis2real, label='Distance to true x')
        ax.plot(sparsity_levels_0, label='Sparsity_Level_0')
        ax.plot(sparsity_levels_1, label='Sparsity_Level_1')
        ax.plot(sparsity_levels_2, label='Sparsity_Level_2')
        ax.plot(sparsity_levels_3, label='Sparsity_Level_3')
        ax.set_title(f'Lambda = {lambda_val}')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Distance')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()
