from sympy import diff, symbols, cos, sin, ln
import numpy as np
import math
from urllib.request import urlopen
import pandas as pd
import urllib
from urllib import request

# 2.7.1
# def vectorized_forward_pass(self, input_matrix):
#     """
#     Метод рассчитывает ответ перцептрона при предъявлении набора примеров
#     input_matrix - матрица примеров размера (n, m), каждая строка - отдельный пример,
#     n - количество примеров, m - количество переменных
#     Возвращает вертикальный вектор размера (n, 1) с ответами перцептрона
#     (элементы вектора - boolean или целые числа (0 или 1))
#     """
#     return np.where(input_matrix.dot(self.w) + self.b > 0, 1, 0)

# 2.7.2
# def train_on_single_example(self, example, y):
#     """
#     принимает вектор активации входов example формы (m, 1)
#     и правильный ответ для него (число 0 или 1 или boolean),
#     обновляет значения весов перцептрона в соответствии с этим примером
#     и возвращает размер ошибки, которая случилась на этом примере до изменения весов (0 или 1)
#     (на её основании мы потом построим интересный график)
#     """
#     pred = float(self.w.T.dot(example) + self.b) > 0
#     oshibka = y - pred
#     self.w = self.w + oshibka * example
#     self.b = self.b + oshibka
#     return oshibka

# 2.7.3
# def summatory(self, input_matrix):
#     """
#     Вычисляет результат сумматорной функции для каждого примера из input_matrix.
#     input_matrix - матрица примеров размера (n, m), каждая строка - отдельный пример,
#     n - количество примеров, m - количество переменных.
#     Возвращает вектор значений сумматорной функции размера (n, 1).
#     """
#     self.summatory_activation = input_matrix.dot(self.w)
#     return self.summatory_activation
#
# def activation(self, summatory_activation):
#     """
#     Вычисляет для каждого примера результат активационной функции,
#     получив на вход вектор значений сумматорной функций
#     summatory_activation - вектор размера (n, 1),
#     где summatory_activation[i] - значение суммматорной функции для i-го примера.
#     Возвращает вектор размера (n, 1), содержащий в i-й строке
#     значение активационной функции для i-го примера.
#     """
#     return self.activation_function(summatory_activation)
#
# def vectorized_forward_pass(self, input_matrix):
#     """
#     Векторизованная активационная функция логистического нейрона.
#     input_matrix - матрица примеров размера (n, m), каждая строка - отдельный пример,
#     n - количество примеров, m - количество переменных.
#     Возвращает вертикальный вектор размера (n, 1) с выходными активациями нейрона
#     (элементы вектора - float)
#     """
#     return self.activation(self.summatory(input_matrix))

# 2.7.4
# def J_quadratic(neuron, X, y):
#     """
#     Оценивает значение квадратичной целевой функции.
#     Всё как в лекции, никаких хитростей.
#
#     neuron - нейрон, у которого есть метод vectorized_forward_pass, предсказывающий значения на выборке X
#     X - матрица входных активаций (n, m)
#     y - вектор правильных ответов (n, 1)
#
#     Возвращает значение J (число)
#     """
#
#     assert y.shape[1] == 1, 'Incorrect y shape'
#
#     return 0.5 * np.mean((neuron.vectorized_forward_pass(X) - y) ** 2)
#
# def J_quadratic_derivative(y, y_hat):
#     """
#     Вычисляет вектор частных производных целевой функции по каждому из предсказаний.
#     y_hat - вертикальный вектор предсказаний,
#     y - вертикальный вектор правильных ответов,
#
#     В данном случае функция смехотворно простая, но если мы захотим поэкспериментировать
#     с целевыми функциями - полезно вынести эти вычисления в отдельный этап.
#
#     Возвращает вектор значений производной целевой функции для каждого примера отдельно.
#     """
#     assert y_hat.shape == y.shape and y_hat.shape[1] == 1, 'Incorrect shapes'
#     return (y_hat - y) / len(y)
#
# def compute_grad_analytically(neuron, X, y, J_prime=J_quadratic_derivative):
#     """
#     Аналитическая производная целевой функции
#     neuron - объект класса Neuron
#     X - вертикальная матрица входов формы (n, m), на которой считается сумма квадратов отклонений
#     y - правильные ответы для примеров из матрицы X
#     J_prime - функция, считающая производные целевой функции по ответам
#
#     Возвращает вектор размера (m, 1)
#     """
#     # Вычисляем активации
#     # z - вектор результатов сумматорной функции нейрона на разных примерах
#     z = neuron.summatory(X)
#     y_hat = neuron.activation(z)
#     # Вычисляем нужные нам частные производные
#     dy_dyhat = J_prime(y, y_hat)
#     dyhat_dz = neuron.activation_function_derivative(z)
#     # осознайте эту строчку:
#     dz_dw = X
#     # а главное, эту:
#     grad = ((dy_dyhat * dyhat_dz).T).dot(dz_dw)
#     # можно было написать в два этапа. Осознайте, почему получается одно и то же
#     # grad_matrix = dy_dyhat * dyhat_dz * dz_dw
#     # grad = np.sum(, axis=0)
#     # Сделаем из горизонтального вектора вертикальный
#     grad = grad.T
#     return grad
#
# def update_mini_batch(self, X, y, learning_rate, eps):
#     """
#     X - матрица размера (batch_size, m)
#     y - вектор правильных ответов размера (batch_size, 1)
#     learning_rate - константа скорости обучения
#     eps - критерий остановки номер один: если разница между значением целевой функции
#     до и после обновления весов меньше eps - алгоритм останавливается.
#
#     Рассчитывает градиент (не забывайте использовать подготовленные заранее внешние функции)
#     и обновляет веса нейрона. Если ошибка изменилась меньше, чем на eps - возвращаем 1,
#     иначе возвращаем 0.
#     """
#     f_old = J_quadratic(self, X, y)
#     grad = compute_grad_analytically(self,X,y)
#     self.w -=  learning_rate * grad
#     f_new = J_quadratic(self, X, y)
#     return 1 if f_old - f_new < eps else 0

# 2.7.5
# def SGD(self, X, y, batch_size, learning_rate=0.1, eps=1e-6, max_steps=200):
#     """
#     Внешний цикл алгоритма градиентного спуска.
#     X - матрица входных активаций (n, m)
#     y - вектор правильных ответов (n, 1)
#
#     learning_rate - константа скорости обучения
#     batch_size - размер батча, на основании которого
#     рассчитывается градиент и совершается один шаг алгоритма
#
#     eps - критерий остановки номер один: если разница между значением целевой функции
#     до и после обновления весов меньше eps - алгоритм останавливается.
#     Вторым вариантом была бы проверка размера градиента, а не изменение функции,
#     что будет работать лучше - неочевидно. В заданиях используйте первый подход.
#
#     max_steps - критерий остановки номер два: если количество обновлений весов
#     достигло max_steps, то алгоритм останавливается
#
#     Метод возвращает 1, если отработал первый критерий остановки (спуск сошёлся)
#     и 0, если второй (спуск не достиг минимума за отведённое время).
#     """
#     count = 0
#     res = 1
#     xcom = np.column_stack((X, y))
#     while (count < max_steps):
#         np.random.shuffle(xcom)
#         samplex = xcom[0:batch_size, : -1]
#         sampley = np.array(list([el[-1]] for el in xcom[0:batch_size]))
#         res = self.update_mini_batch(samplex,sampley,learning_rate,eps)
#         if res:
#             return 1
#         count += 1
#     return res

# 2.7.6
# def compute_grad_numerically_2(neuron, X, y, J=J_quadratic, eps=10e-2):
#     """
#     Численная производная целевой функции.
#     neuron - объект класса Neuron с вертикальным вектором весов w,
#     X - вертикальная матрица входов формы (n, m), на которой считается сумма квадратов отклонений,
#     y - правильные ответы для тестовой выборки X,
#     J - целевая функция, градиент которой мы хотим получить,
#     eps - размер $\delta w$ (малого изменения весов).
#     """
#     w_0 = neuron.w.copy()
#     weights = len(w_0)
#     grad = np.zeros(w_0.shape)
#     for i in range(weights):
#         neuron.w = w_0.copy()
#         neuron.w[i] += eps
#         J_p = J(neuron, X, y)
#         neuron.w = w_0.copy()
#         neuron.w[i] -= eps
#         J_m = J(neuron, X, y)
#         grad[i] = (J_p - J_m) / (2 * eps)
#         neuron.w = w_0.copy()
#
#     return grad