import numpy as np
import matplotlib.pylab as plt
from data_generator import generate_data


class ProximalGradientDescent:
    def __init__(self, alpha, lambda_val, stop=1e-5, max_iter=100000):
        self.alpha = alpha
        self.lambda_val = lambda_val
        self.stop = stop
        self.max_iter = max_iter

    def gradient_step(self, A, b, x_k):
        gradient = np.sum([A_i.T @ ((A_i @ x_k) - b_i)
                          for A_i, b_i in zip(A, b)], axis=0)
        return x_k - self.alpha * gradient

    def soft_thresholding_1(self, x_k_half):
        x_k_half = np.sign(
            x_k_half) * np.maximum((np.abs(x_k_half)-self.alpha*self.lambda_val), 0)
        return x_k_half

    def check_convergence(self, x_k, x_k_ud):
        return np.linalg.norm((x_k_ud - x_k), 2) < self.stop

    def fit(self, A, b, x_true):
        Xk = np.zeros_like(x_true)
        X_opt_step = np.zeros(
            (self.max_iter, x_true.shape[0], x_true.shape[1]))
        count = 0
        while count < self.max_iter:
            Xk_half = self.gradient_step(A, b, Xk)
            Xk_ud = self.soft_thresholding_1(Xk_half)
            #print(Xk_ud.shape)
            print("Iter ", count+1, ":", np.linalg.norm((Xk_ud - x_true), 2))

            X_opt_step[count] = Xk_ud
            if self.check_convergence(Xk, Xk_ud):
                print("DONE")
                break

            Xk = Xk_ud
            count += 1

        return X_opt_step[:count + 1,:]


if __name__ == '__main__':
    n = 200
    m = 5
    sparsity = 0.05
    seed = 42
    alpha = 0.001

    x_true, A, e, b = generate_data(n, m, sparsity, seed=seed)
    lambda_vals = [0.001,  0.01, 0.05, 0.1, 1, 10,20, 50]

    fig, axes = plt.subplots(2, 4, figsize=(15, 10))
    axes = axes.flatten()

    for i, lambda_val in enumerate(lambda_vals):
        
        pgd = ProximalGradientDescent(alpha=alpha, lambda_val=lambda_val)
        X_opt_step = pgd.fit(A, b, x_true)

        X_best = X_opt_step[-1]

        X_dis2best = [np.linalg.norm(X_opt - X_best) for X_opt in X_opt_step]
        X_dis2real = [np.linalg.norm(
            X_opt - x_true) for X_opt in X_opt_step]
        sparsity_levels = [(np.sum(X_opt == 0) / len(X_best)) for X_opt in X_opt_step]
        ax = axes[i]
        ax.plot(X_dis2best, label='Distance to X_best')
        ax.plot(X_dis2real, label='Distance to true x')
        ax.plot(sparsity_levels, label='Sparsity_Level')
        ax.set_title(f'Lambda = {lambda_val}')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Distance')
        ax.legend()
        ax.grid(True)


    plt.tight_layout()
    plt.show()

