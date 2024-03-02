import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def GD_SingleVariable(X, y, epochs=1000, stop_condition=0.1, theta_0=0, theta_1=0, lr=0.01):
    loss = []

    for i in tqdm(range(epochs)):
        # print(f'****************** Iteration {i} ********************\n')

        h_x = theta_0 + theta_1 * X
        # print('h(x) : ', h_x)

        error_vector = (h_x - y)
        # print('\nError Vector :\n', error_vector)

        MSE = np.sum(error_vector ** 2) / (2 * len(y))
        loss.append(MSE)
        # print('\nj = ', MSE)

        d_theta_0 = (np.sum(h_x - y)) / len(y)
        d_theta_1 = (np.sum((h_x - y) * X)) / len(y)

        gradient_vector = np.array([d_theta_0, d_theta_1])
        # print('\nGradient Vector : \n', gradient_vector)

        gradient_vec_norm = np.linalg.norm(gradient_vector)
        # print('\n Gradient Vector Norm : ', gradient_vec_norm)

        if gradient_vec_norm <= stop_condition:
            print('****************** Training Report ********************')
            print(f'Gradient Descent converged after {i} iterations')
            print('theta_0_Opt : ', theta_0)
            print('theta_1_Opt : ', theta_1)
            # print('\nError Vector :\n', error_vector)
            print('Cost = ', MSE)
            # print('h(x) = y_predict: \n', h_x)
            # print('y_actual : ', y)
            break

        theta_0 = theta_0 - d_theta_0 * lr
        theta_1 = theta_1 - d_theta_1 * lr
        # print('theta_0_new : ', theta_0)
        # print('theta_1_new : ', theta_1)

    return theta_0, theta_1, loss


def plot_loss(loss):
    plt.figure(figsize=(10, 5))
    sns.lineplot(loss, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss');




#f(x) = 22*x - 5
np.random.seed(42)
a = np.random.random(100)
b = 22*a - 5
theta_0, theta_1, loss = GD_SingleVariable(a, b, epochs=2000, lr=0.5, stop_condition=0.01)
plot_loss(loss)