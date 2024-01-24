import numpy as np


def generate_data(x_dim, b_dim, sparsity, seed=42):
    np.random.seed(seed)

    x_true = np.zeros((x_dim, 1))
    non_zero_indices = np.random.choice(
        x_dim, int(sparsity * x_dim), replace=False)
    x_true[non_zero_indices, 0] = np.random.randn(int(sparsity * x_dim))
    # generate A_i
    A = [np.random.randn(b_dim, x_dim) for _ in range(10)]

    # generate noise e_i
    e = [np.random.randn(b_dim, 1) * np.sqrt(0.1) for _ in range(10)]

    # generate b_i
    b = [A_i @ x_true + e_i for A_i, e_i in zip(A, e)]

    return x_true, A, e, b


def check_sparsity_and_statistics(x, expected_sparsity):
    actual_sparsity = np.sum(x != 0) / x.size
    mean = np.mean(x[x != 0])
    variance = np.var(x[x != 0], ddof=1)
    return actual_sparsity, mean, variance


if __name__ == '__main__':

    x_dim = 200
    b_dim = 5
    sparsity = 0.05

    # Generate data
    x_true, A, e, b = generate_data(x_dim, b_dim, sparsity)
    print(x_true.shape, np.array(A).shape,
          np.array(e).shape, np.array(b).shape)


    actual_sparsity = 0
    means = 0
    variances = 0
    a_means = 0
    a_vars = 0
    e_means = 0
    e_vars = 0
    tot = 1000
    for seed in range(0,tot):
        x_true, A, e, b = generate_data(x_dim, b_dim, sparsity,seed=seed)
        actual_sparsity, mean, variance = check_sparsity_and_statistics(
        x_true, sparsity)
        
        A_flat = np.array(A).flatten()
        e_flat = np.array(e).flatten()
        actual_sparsity = actual_sparsity
        means +=mean
        variances += variance
        a_means += np.mean(A_flat)
        a_vars += np.var(A_flat)
        e_means += np.mean(e_flat)
        e_vars += np.var(e_flat)

    print(f"Actual Sparsity: {actual_sparsity/tot}")
    print(f"Mean of non-zero elements in x_true: {means/tot}")
    print(f"Variance of non-zero elements in x_true: {variances/tot}")
    print(f"Mean of A: {a_means/tot}, Variance of A: {a_vars/tot}")
    print(f"Mean of e: {e_means/tot}, Variance of e: {e_vars/tot}")