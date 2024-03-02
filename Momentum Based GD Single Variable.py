import numpy as np
from tqdm import tqdm

def Momentum_GD_SingleVariable(X, y, alpha, max_num_iterations, gamma):
    theta_0, theta_1 = 0 ,0
    loss = []
    theta_0s = []
    theta_1s = []
    h_xs  =[]
    m = len(y)
    v0, v1 = 0 ,0

    for i in tqdm(range(max_num_iterations)):
        # print(f'****************** Iteration {i} ********************\n')

        theta_0s.append(theta_0)
        theta_1s.append(theta_1)

        h_x = theta_0 +theta_1 *X
        h_xs.append(h_x)
        # print('h(x) : ', h_x)

        error_vector = (h_x - y)
        # print('\nError Vector :\n', error_vector)

        MSE = np.sum(error_vector**2) / (2*m)
        loss.append(MSE)
        # print('\nj = ', MSE)

        d_theta_0 = (np.sum(h_x - y) ) /m
        d_theta_1 = (np.sum((h_x - y) *X)) / m

        gradient_vector = np.array([d_theta_0, d_theta_1])
        # print('\nGradient Vector : \n', gradient_vector)

        gradient_vec_norm = np.linalg.norm(gradient_vector)
        # print('\n Gradient Vector Norm : ', gradient_vec_norm)

        if i > 0:
            if gradient_vec_norm <= 0.001 or (np.abs(loss[-1] - loss[-2])) <= 0.001:
                print('****************** Training Report ********************')
                print(f'Gradient Descent converged after {i} iterations')
                print('theta_0_Opt : ', theta_0)
                print('theta_1_Opt : ', theta_1)
                # print('\nError Vector :\n', error_vector)
                print('Cost = ', MSE)
                # print('h(x) = y_predict: \n', h_x)
                # print('y_actual : ', y)

                break

        v0 = v0 * gamma + d_theta_0 * alpha
        v1 = v1 * gamma + d_theta_1 * alpha

        theta_0 = theta_0 - v0
        theta_1 = theta_1 - v1
    #         print('theta_0_new : ', theta_0)
    #         print('theta_1_new : ', theta_1)

    return loss, theta_0s, theta_1s, h_xs


np.random.seed(42)
a = np.random.random(100)
b = 22*a - 5
loss, theta_0s, theta_1s, h_xs = Momentum_GD_SingleVariable(a, b, alpha=0.5, max_num_iterations=2000, gamma=0.8)