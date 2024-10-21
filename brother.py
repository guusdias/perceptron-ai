import numpy as np
import math

def sigmoid(u):
    return (2 / (1 + np.exp(-u))) - 1

def sigmoid_derivative(o):
    return (1 - o * o) / 2

def findOutput(data, w):
    lamb = 0.10
    u = np.dot(data, w)  
    return sigmoid(lamb * u)

p = np.array([[1, 1, -1], [1, -1, -1], [-1, 1, -1], [-1, -1, -1]])
d = np.array([1, 1, 1, -1])  

w = np.random.rand(p.shape[1])

c = 0.5  
d_error = 0.01  
max_iter = 10000  

iter = 0
while True:
    error = 0
    for i in range(len(p)):
        o = findOutput(p[i], w) 
        error += 0.5 * (d[i] - o) ** 2.0 
        delta = (d[i] - o) * sigmoid_derivative(o)  

        for k in range(len(w)):
            w[k] += c * delta * p[i][k]

    iter += 1
    print(f"Iteração {iter}, Erro: {error}, Pesos: {w}")

    if error < d_error or iter >= max_iter:
        print(f"Treinamento concluído após {iter} iterações.")
        break

print("Testes com a rede treinada (OR)")
print(findOutput([1, 1, -1], w))  # Deve estar próximo de 1
print(findOutput([1, -1, -1], w))  # Deve estar próximo de 1
print(findOutput([-1, 1, -1], w))  # Deve estar próximo de 1
print(findOutput([-1, -1, -1], w))  # Deve estar próximo de -1
