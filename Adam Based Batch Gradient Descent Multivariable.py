import numpy as np
from tqdm import tqdm
from sklearn.metrics import r2_score

def Adam(X, y, alpha, beta1, beta2, epsilon=10e-8, epochs=300):
    theta_0s, theta_1s = [], []
    all_hx, loss = [], []
    m = len(y)
    y = y.reshape(-1,1)

    ones = np.ones((X.shape[0])).reshape(-1, 1)
    X = np.concatenate((ones, X), axis=1)
    dim = X.shape[1]

    thetas = np.zeros(dim).reshape(-1,1)
    V = np.zeros(dim).reshape(-1,1)
    M = np.zeros(dim).reshape(-1,1)


    for i in tqdm(range(epochs)):
        #         print(f'****************** Iteration {i} ********************\n')
        h_x = X@thetas
        all_hx.append(h_x)
        #         print('h(x) : ', h_x)

        error_vector = h_x - y

        MSE = np.sum(error_vector ** 2) / (2 * m)
        loss.append(MSE)
        #         print('\nj = ', MSE)

        d_thetas = (X.T @ error_vector) / m

        gradient_vec_norm = np.linalg.norm(d_thetas)
        #         print('\n Gradient Vector Norm : ', gradient_vec_norm)

        if i > 0:
            if gradient_vec_norm <= 0.001 or (np.abs(loss[-1] - loss[-2])) <= 0.001:
                print('****************** Training Report ********************')
                print(f'Gradient Descent converged after {i} iterations')
                print('thetas : ', thetas)
                print('Cost = ', MSE)

                break

        t = i + 1

        M = beta1 * M + (1 - beta1) * d_thetas
        V = V * beta2 + (1 - beta2) * (d_thetas**2)

        # Bais Correction
        m_hat = M / (1 - (beta1 ** t))
        v_hat = V / (1 - (beta2 ** t))

        # Updating Thetas
        thetas = thetas - (alpha * m_hat) / (np.sqrt(v_hat) + epsilon)

    return thetas, loss, all_hx


np.random.seed(42)
a = np.random.random((100, 5))
b = 22*a[:,0] + 3*a[:, 1] + a[:, 2] - 10*a[:, 3] + 100*a[:, 4] - 5
thetas, loss, all_hx = Adam(a, b, alpha=0.4, beta1=0.9, beta2=0.999, epochs=2000)
print(r2_score(b, all_hx[-1]))