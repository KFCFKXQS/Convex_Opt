import numpy as np
import matplotlib.pyplot as plt
from data_generator import generate_data

# ADMM Algorithm Implementation


class ADMM:
    def __init__(self, rho, lamda, x_dim, b_dim, stop=1e-5, max_iter=100000):
        self.rho = rho
        self.lamda = lamda
        self.stop = stop
        self.max_iter = max_iter
        self.x_dim = x_dim
        self.b_dim = b_dim

    def update_x(self, A, b, Y, T):
        A_T = self.rho * np.eye(self.x_dim)
        A_b = np.zeros((self.x_dim, 1))

        for i in range(10):
            A_T += A[i].T @ A[i]
            A_b += A[i].T @ b[i]

        A_T_inv = np.linalg.inv(A_T)
        X_update = A_T_inv @ (A_b + self.rho * Y - T)
        return X_update

    def update_y(self, X, T):
        Y_update = np.sign(X + T / self.rho) * np.maximum(np.abs(X +
                                                                 T / self.rho) - self.lamda / self.rho, 0)
        return Y_update

    def update_t(self, T, X, Y):
        T_update = T + self.rho * (X - Y)
        return T_update

    def check_convergence(self, x_k, x_k_ud):
        return np.linalg.norm((x_k_ud - x_k), 2) < self.stop

    def fit(self, A, b, x_true):
        # 初始化 X, Y, T
        X = np.zeros((self.x_dim, 1))
        Y = np.zeros((self.x_dim, 1))
        T = np.zeros((self.x_dim, 1))
        count = 0
        X_opt_step = np.zeros(
            (self.max_iter, x_true.shape[0], x_true.shape[1]))

        while count < self.max_iter:
            X_update = self.update_x(A, b, Y, T)
            Y_update = self.update_y(X_update, T)
            T_update = self.update_t(T, X_update, Y_update)

            X_opt_step[count] = X_update
            print("Iter ", count+1, ":", np.linalg.norm((X_update - x_true), 2))
            if self.check_convergence(X_update, X):
                break

            # 更新 X, Y, T
            X, Y, T = X_update, Y_update, T_update
            count += 1
        return X_opt_step[:count + 1, :]


if __name__ == '__main__':
    n = 200
    m = 5
    sparsity = 0.05
    seed = 42
    rho = 0.005
    x_true, A, e, b = generate_data(n, m, sparsity, seed=seed)
    lambda_vals = [0.00001,0.0001,0.0005,0.001, 0.05, 0.1]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, lambda_val in enumerate(lambda_vals):

        admm = ADMM(rho=rho, lamda=lambda_val, x_dim=n, b_dim=m)
        X_opt_step = admm.fit(A, b, x_true)

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
        ax = axes[i]
        ax.plot(X_dis2best, label='Distance to X_best', linewidth=3)
        ax.plot(X_dis2real, label='Distance to true x', linewidth=1.5)
        ax.plot(sparsity_levels_0, label='Sparsity_Level_0')
        ax.plot(sparsity_levels_1, label='Sparsity_Level_1')
        ax.plot(sparsity_levels_2, label='Sparsity_Level_2')
        ax.set_title(f'Lambda = {lambda_val}')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Distance')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()
