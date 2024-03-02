import numpy as np

def Gradient_descent(func, first_grad, x, epsilon, alpha, epochs):
    for i in range(epochs):
        gradient = first_grad(x)
        if abs(gradient) < epsilon:
            break

        x = x - alpha * gradient

    print(f'GD Results with x0 ={x}, lr={alpha}, epsilon={epsilon}')
    print(f'Found solution after {i} iterations.')
    print(f'x_min = {x}')
    print(f'Gradient = {first_grad(x)}')

    return x, first_grad(x)


def Newton_method(func, first_grad, second_grad, x, epsilon, epochs):
    for i in range(epochs):
        gradient = first_grad(x)

        if abs(gradient) < epsilon:
            break

        second_gradient = second_grad(x)
        x = x - (gradient / second_gradient)

    print(f'Newton Results with x0 ={x}, epsilon={epsilon}')
    print(f'Found solution after {i} iterations.')
    print(f'x_min = {x}')
    print(f'Gradient = {first_grad(x)}')

    return x, first_grad(x)


def Newton_method_withAlpha(func, first_grad, second_grad, x, epsilon, epochs, alpha):
    for i in range(epochs):
        gradient = first_grad(x)

        if abs(gradient) < epsilon:
            break

        second_gradient = second_grad(x)
        x = x - alpha * (gradient / second_gradient)

    print(f'Newton Results with x0 ={x}, epsilon={epsilon}')
    print(f'Found solution after {i} iterations.')
    print(f'x_min = {x}')
    print(f'Gradient = {first_grad(x)}')

    return x, first_grad(x)


np.random.seed(42)
a = np.random.random(100)
b = 5*a**2 + 3*a - 12

func = lambda a: 5*a**2 + 3*a - 12
first_grad = lambda a: 10*a + 3
second_grad = lambda a: 10

Gradient_descent(func, first_grad, x=40, epsilon=0.1, alpha=0.01, epochs=350)
print('\n')
Newton_method(func, first_grad, second_grad, x=40, epsilon=0.1, epochs=350)
print('\n')
