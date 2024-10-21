import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=100, desired_error=0.01, max_iter=1000):
        self.weights = np.random.randn(input_size + 1) * 0.01  
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.desired_error = desired_error
        self.max_iter = max_iter

    def activation_function(self, x):
        return np.where(x >= 0, 1, 0)

    def predict(self, X):
        X = np.insert(X, 0, -1, axis=1)
        weighted_sum = np.dot(X, self.weights)
        return self.activation_function(weighted_sum)

    def train(self, X, y):
        X = np.insert(X, 0, -1, axis=1)
        error_list = []
        for epoch in range(self.epochs):
            total_error = 0
            for i in range(X.shape[0]):
                y_pred = self.activation_function(np.dot(X[i], self.weights))
                error = y[i] - y_pred
                total_error += abs(error)
                self.weights += self.learning_rate * error * X[i]
                print(f"Pesos atualizados: {self.weights}")
            
            error_list.append(total_error)
            print(f"Epoch {epoch+1}/{self.epochs}, Erro Total: {total_error}")
            
            if total_error <= self.desired_error:
                print(f"Convergência atingida após {epoch+1} épocas.")
                break

            if epoch >= self.max_iter:
                print(f"Limite de iterações atingido: {self.max_iter}")
                break
        
        plt.plot(range(len(error_list)), error_list)
        plt.xlabel('Épocas')
        plt.ylabel('Erro Total')
        plt.title('Erro em relação às iterações')
        plt.show()

X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1]) 

X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_or = np.array([0, 1, 1, 1])  

learning_rate = 0.1  
desired_error = 0.01 
max_iter = 1000  

print("Treinando Perceptron para AND")
perceptron_and = Perceptron(input_size=2, learning_rate=learning_rate, desired_error=desired_error, max_iter=max_iter)
perceptron_and.train(X_and, y_and)

print("Testando Perceptron para AND")
for inputs in X_and:
    print(f"Entrada: {inputs}, Saída: {perceptron_and.predict([inputs])[0]}")

print("\nTreinando Perceptron para OR")
perceptron_or = Perceptron(input_size=2, learning_rate=learning_rate, desired_error=desired_error, max_iter=max_iter)
perceptron_or.train(X_or, y_or)

print("Testando Perceptron para OR")
for inputs in X_or:
    print(f"Entrada: {inputs}, Saída: {perceptron_or.predict([inputs])[0]}")
