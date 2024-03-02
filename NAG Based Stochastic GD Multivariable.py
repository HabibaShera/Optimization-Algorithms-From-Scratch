import numpy as np
from tqdm import tqdm
from sklearn.metrics import r2_score

def NAG(X, y, alpha, max_num_iterations, gamma):
    y = y.reshape(-1,1)
    x0 = np.ones(len(X)).reshape(-1,1)
    X = np.concatenate((x0, X), axis=1)

    thetas = np.zeros(X.shape[1]).reshape(-1,1)
    loss = []
    h_xs = []
    m = len(y)
    V = np.zeros(X.shape[1]).reshape(-1,1)

    for epoch in tqdm(range(max_num_iterations)):
        #         print(f'****************** Iteration {i} ********************\n')
        for i in range(m):
            theta_temp = thetas - gamma * V
            h_x = X[i].reshape(1,-1)@theta_temp
            h_xs.append(h_x)
            #         print('h(x) : ', h_x)

            error_vector = (h_x - y[i])
            #         print('\nError Vector :\n', error_vector)

            MSE = (error_vector ** 2) / (2)
            loss.append(MSE)
            #         print('\nj = ', MSE)

        d_thetas = X[i].reshape(-1,1) * error_vector
        gradient_vec_norm = np.linalg.norm(d_thetas)
            #         print('\n Gradient Vector Norm : ', gradient_vec_norm)

        if epoch > 0:
            if gradient_vec_norm <= 0.001 or (np.abs(loss[-1] - loss[-m])) <= 0.001:
                print('****************** Training Report ********************')
                print(f'Gradient Descent converged after {epoch} iterations')
                print('thetas : ', thetas)
                print('Cost = ', MSE)
                break

        V = V * gamma + d_thetas * alpha
        thetas = theta_temp - alpha * d_thetas

    return loss, thetas, h_xs


np.random.seed(42)
a = np.random.random((100, 5))
b = 22*a[:,0] + 3*a[:, 1] + a[:, 2] - 10*a[:, 3] + 100*a[:, 4] - 5
loss, thetas, h_xs = NAG(a, b, alpha=0.1, max_num_iterations=2000, gamma=0.999)

a_ = np.concatenate((np.ones(len(a)).reshape(-1,1), a), axis=1)
hx = a_@thetas
print(r2_score(b, hx))

