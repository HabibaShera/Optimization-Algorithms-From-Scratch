import numpy as np
from tqdm import tqdm


def RMSProp(X, y, alpha, beta, epsilon=10e-8, epochs=300):
    theta_0s, theta_1s = [], []
    all_hx, loss = [], []
    theta_0, theta_1 = 0, 0
    v0, v1 = 0, 0
    m = len(y)

    for i in range(epochs):
        #         print(f'****************** Iteration {i} ********************\n')
        theta_0s.append(theta_0)
        theta_1s.append(theta_1)

        h_x = theta_0 + theta_1 * X
        all_hx.append(h_x)
        #         print('h(x) : ', h_x)

        error_vector = h_x - y

        MSE = np.sum(error_vector ** 2) / (2 * m)
        loss.append(MSE)
        #         print('\nj = ', MSE)

        d_theta_0 = (np.sum(h_x - y)) / m
        d_theta_1 = (np.sum((h_x - y) * X)) / m

        gradient_vector = np.array([d_theta_0, d_theta_1])
        #         print('\nGradient Vector : \n', gradient_vector)

        gradient_vec_norm = np.linalg.norm(gradient_vector)
        #         print('\n Gradient Vector Norm : ', gradient_vec_norm)

        if i > 0:
            if gradient_vec_norm <= 0.001 or (np.abs(loss[-1] - loss[-2])) <= 0.001:
                print('****************** Training Report ********************')
                print(f'Gradient Descent converged after {i} iterations')
                print('theta_0_Opt : ', theta_0)
                print('theta_1_Opt : ', theta_1)
                #                 print('\nError Vector :\n', error_vector)
                print('Cost = ', MSE)
                #                 print('h(x) = y_predict: \n', h_x)
                #                 print('y_actual : ', y)

                break

        v0 = beta * v0 + (1 - beta) * d_theta_0 ** 2
        v1 = beta * v1 + (1 - beta) * d_theta_1 ** 2

        theta_0 = theta_0 - (alpha * d_theta_0) / (np.sqrt(v0) + epsilon)
        theta_1 = theta_1 - (alpha * d_theta_1) / (np.sqrt(v1) + epsilon)

    return theta_0s, theta_1s, loss, all_hx


np.random.seed(42)
a = np.random.random(100)
b = 22*a - 5
theta_0s, theta_1s, loss, all_hx = RMSProp(a, b, alpha=0.5, beta=0.9, epochs=2000)