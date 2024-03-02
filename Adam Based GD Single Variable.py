import numpy as np
from tqdm import  tqdm


def Adam(X, y, alpha, beta1, beta2, epsilon=10e-8, epochs=300):
    theta_0s, theta_1s = [], []
    all_hx, loss = [], []
    theta_0, theta_1 = 0, 0
    v0, v1 = 0, 0
    m0, m1 = 0, 0
    m = len(y)

    for i in tqdm(range(epochs)):
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
        t = i + 1
        m0 = beta1 * m0 + (1 - beta1) * d_theta_0
        m1 = beta1 * m1 + (1 - beta1) * d_theta_1

        v0 = v0 * beta2 + (1 - beta2) * (d_theta_0 ** 2)
        v1 = v1 * beta2 + (1 - beta2) * (d_theta_1 ** 2)

        # Bais Correction
        m_hat_0 = m0 / (1 - (beta1 ** t))
        v_hat_0 = v0 / (1 - (beta2 ** t))

        m_hat_1 = m1 / (1 - (beta1 ** t))
        v_hat_1 = v1 / (1 - (beta2 ** t))

        # Updating Thetas
        theta_0 = theta_0 - (alpha * m_hat_0) / (np.sqrt(v_hat_0) + epsilon)
        theta_1 = theta_1 - (alpha * m_hat_1) / (np.sqrt(v_hat_1) + epsilon)

    return theta_0s, theta_1s, loss, all_hx



np.random.seed(42)
a = np.random.random(100)
b = 22*a - 5
theta_0s, theta_1s, loss, all_hx = Adam(a, b, alpha=0.4, beta1=0.9, beta2=0.999, epochs=2000)