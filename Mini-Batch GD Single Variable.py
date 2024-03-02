import numpy as np
from tqdm import tqdm


def Mini_Batch_GD_SingleVariable(X, y, alpha, epochs, batch_size):
    theta_0, theta_1 = 0, 0
    m = len(y)

    losses, thetas_0, thetas_1 = [], [], []
    for epoch in tqdm(range(epochs)):
        # print(f'****************** Epoch {epoch} ********************\n')

        for i in range(0, m, batch_size):
            thetas_0.append(theta_0)
            thetas_1.append(theta_1)

            h_x = theta_0 + theta_1 * X[i:i + batch_size]
            error_vector = (h_x - y[i:i + batch_size])

            MSE = sum(error_vector ** 2) / (2) * batch_size

            d_theta_0 = sum(h_x - y[i:i + batch_size]) / batch_size
            d_theta_1 = sum((h_x - y[i:i + batch_size]) * X[i:i + batch_size]) / batch_size

            theta_0 = theta_0 - d_theta_0 * alpha
            theta_1 = theta_1 - d_theta_1 * alpha

            losses.append(MSE)

        if epoch == 0:
            num = len(losses)

        gradient_vector = np.array([d_theta_0, d_theta_1])
        gradient_vec_norm = np.linalg.norm(gradient_vector)
        if epoch > 0:
            if (gradient_vec_norm <= 0.001) or (np.abs(losses[-1] - losses[len(losses) - num]) <= 0.001):
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
losses, thetas_0, thetas_1 = Mini_Batch_GD_SingleVariable(a, b, epochs=2000, alpha=0.5, batch_size=16)