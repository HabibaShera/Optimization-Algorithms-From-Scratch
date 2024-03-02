import numpy as np

def Gradient_descent_Multi_variables(func, first_grad, x, epsilon, alpha, epochs):
    x = np.array(x)
    for i in range(epochs):
        gradient = first_grad(*x)
        if np.linalg.norm(gradient) < epsilon:
            break

        x = x - alpha * gradient

    print(f'GD Results with x0 ={x}, lr={alpha}, epsilon={epsilon}')
    print(f'Found solution after {i} iterations.')
    print(f'x_min = {x}')
    print(f'Gradient = {first_grad(*x)}')

    return x, first_grad(*x)


def Newton_method_Multi_variables(func, first_grad, second_grad, x, epsilon, epochs):
    x = np.array(x)
    for i in range(epochs):
        gradient = first_grad(*x)
        if np.linalg.norm(gradient) < epsilon:
            break

        second_gradient = second_grad(*x)

        x = x - (np.linalg.inv(second_gradient) @ gradient)

    print(f'Newton Results with x0 ={x}, epsilon={epsilon}')
    print(f'Found solution after {i} iterations.')
    print(f'x_min = {x}')
    print(f'Gradient = {first_grad(*x)}')

    return x, first_grad(*x)


def Newton_method_Multi_variables_withAlpha(func, first_grad, second_grad, x, epsilon, epochs, alpha):
    x = np.array(x)
    for i in range(epochs):
        gradient = first_grad(*x)
        if np.linalg.norm(gradient) < epsilon:
            break

        second_gradient = second_grad(*x)

        x = x - alpha * (np.linalg.inv(second_gradient) @ gradient)

    print(f'Newton Results with x0 ={x}, epsilon={epsilon}')
    print(f'Found solution after {i} iterations.')
    print(f'x_min = {x}')
    print(f'Gradient = {first_grad(*x)}')

    return x, first_grad(*x)