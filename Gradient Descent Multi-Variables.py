import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def GD_MultiVariables(X, y, theta=None, epochs=1000, stop_condition=0.1, lr=0.01):
    loss = []
    m = len(y)

    x_0 = np.ones(len(X))
    X = np.concatenate((x_0[:, np.newaxis], X), axis=1)
    y = y.reshape(-1, 1)

    if theta is None:
        theta = np.zeros(X.shape[1]).reshape(-1, 1)

    for i in tqdm(range(epochs)):
        # print(f'****************** Iteration {i} ********************')
        h_x = X @ theta
        # print('h(x):', h_x)

        error_vector = h_x - y
        # print('\nError Vector : ', error_vector)

        MSE = np.sum(error_vector ** 2) / (2 * m)
        loss.append(MSE)
        # print('\nj = ', MSE)

        gradient_vector = (X.T @ error_vector) / m
        # print('\nGradient Vector =  ', gradient_vector)

        gradient_vec_norm = np.linalg.norm(gradient_vector)
        # print('\nGradient vector norm = ', gradient_vec_norm)

        if gradient_vec_norm < stop_condition:
            break

        theta = theta - lr * gradient_vector
    print('\nTheta news :\n', theta)
    return theta, loss


def plot_loss(loss):
    plt.figure(figsize=(10, 5))
    sns.lineplot(loss, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss');


# 10*x1 + 1.5*x2 - 2*x3 - 1
a = np.random.random((100,3))
b = 10*a[:,0] + 1.5*a[:,1] - 2*a[:, 2] - 1

theta, loss = GD_MultiVariables(a, b, lr=0.5, stop_condition=0.01, epochs=2000)
plot_loss(loss)