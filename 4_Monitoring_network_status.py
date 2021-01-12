import numpy as np

# 4.4.1
# def backprop(self, x, y):
#     """
#     Возвращает кортеж ``(nabla_b, nabla_w)`` -- градиент целевой функции по всем параметрам сети.
#     ``nabla_b`` и ``nabla_w`` -- послойные списки массивов ndarray,
#     такие же, как self.biases и self.weights соответственно.
#     """
#
#     nabla_b = [np.zeros(b.shape) for b in self.biases]
#     nabla_w = [np.zeros(w.shape) for w in self.weights]
#
#     # прямое распространение (forward pass)
#     activation = x
#     activations = [x]  # список для послойного хранения активаций
#     zs = []  # список для послойного хранения z-векторов
#     for b, w in zip(self.biases, self.weights):
#         z = np.dot(w, activation) + b
#         zs.append(z)
#         activation = sigmoid(z)
#         activations.append(activation)
#
#     # обратное распространение (backward pass)
#     delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
#     nabla_b[-1] = delta # производная J по смещениям выходного слоя
#     nabla_w[-1] = np.dot(delta, activations[-2].transpose()) # производная J по весам выходного слоя
#
#     # Обратите внимание, что переменная l в цикле ниже используется
#     # немного иначе, чем в лекциях.  Здесь l = 1 означает последний слой,
#     # l = 2 - предпоследний и так далее.
#     # Мы перенумеровали схему, чтобы с удобством для себя
#     # использовать тот факт, что в Python к переменной типа list
#     # можно обращаться по негативному индексу.
#     for l in range(2, self.num_layers):
#         z = zs[-l]
#         sp = sigmoid_prime(z)
#         delta = np.dot(self.weights[-l+1].transpose(), delta) * sp # ошибка на слое L-l
#         nabla_b[-l] = delta # производная J по смещениям L-l-го слоя
#         nabla_w[-l] = np.dot(delta, activations[-l-1].transpose()) # производная J по весам L-l-го слоя
#     return nabla_b, nabla_w