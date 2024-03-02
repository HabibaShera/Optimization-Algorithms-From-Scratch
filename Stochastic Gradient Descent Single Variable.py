import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def SGD_SingleVariable(X, y, alpha, epochs):
    theta_0, theta_1 = 0, 0
    m = len(y)

    losses, thetas_0, thetas_1 = [], [], []
    for epoch in tqdm(range(epochs)):
        # print(f'****************** Epoch {epoch} ********************\n')

        for i in range(m):
            thetas_0.append(theta_0)
            thetas_1.append(theta_1)

            h_x = theta_0 + theta_1 * X[i]
            error_vector = (h_x - y[i])

            MSE = error_vector ** 2 / (2)

            d_theta_0 = h_x - y[i]
            d_theta_1 = (h_x - y[i]) * X[i]

            theta_0 = theta_0 - d_theta_0 * alpha
            theta_1 = theta_1 - d_theta_1 * alpha

            losses.append(MSE)

        gradient_vector = np.array([d_theta_0, d_theta_1])
        gradient_vec_norm = np.linalg.norm(gradient_vector)
        if epoch > 0:
            if (gradient_vec_norm <= 0.001) or (np.abs(losses[-1] - losses[-m]) <= 0.001):
                print('****************** Training Report ********************\n')

                print(f'Gradient Descent converged after {epoch} epochs')
                print('\ntheta_0_opt : ', theta_0)
                print('theta_1_opt : ', theta_1)
                print('Cost = ', MSE)
                break

        # print('Cost = ', MSE)
        # print('\nGradient Vector : \n', gradient_vector)
        # print('\nGradient Vector Norm : \n', gradient_vec_norm)
        # print('\ntheta_0_new : ', theta_0)
        # print('theta_1_new : ', theta_1)

    return losses, thetas_0, thetas_1


np.random.seed(42)
a = np.random.random(100)
b = 22*a - 5
losses, thetas_0, thetas_1 = SGD_SingleVariable(a, b, epochs=2000, alpha=0.5)