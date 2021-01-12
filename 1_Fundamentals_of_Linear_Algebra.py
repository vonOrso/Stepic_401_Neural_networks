from sympy import diff, symbols, cos, sin, ln
import numpy as np
import math
from urllib.request import urlopen
import pandas as pd
import urllib
from urllib import request

# 1.3.4
# Найти производную
# x = symbols('x')
# print(diff((1/4)*ln(((x**2)-1)/((x**2)+1))))
#
# x = 3
# print(0.25*(x**2 + 1)*(-2*x*(x**2 - 1)/(x**2 + 1)**2 + 2*x/(x**2 + 1))/(x**2 - 1))

# 1.3.5
# a = 25**125
# b = 50**100
# c = 100**50
# d = 125**25
# v = 25**(137)
# print((a*b)/(c*d*v))

# 1.3.6
# print((np.log2(8)+np.log2(18))/(2*np.log2(2)+np.log2(3)))

# 1.3.7
# print(math.log(20,225)*math.log(15,289)*math.log(17,20))

# 1.5.1
# a = np.array([[1,0,0,0],
#               [0,1,0,0],
#               [0,0,1,0],
#               [0,0,0,1]])
#
# b = np.array([[1,2,3],
#               [5,6,7],
#               [9,10,11],
#               [4,8,12]])
#
# c = np.array([[3,0,0],
#               [0,3,0],
#               [0,0,3]])
#
# print((a*2).dot(b).dot(c))

# 1.6.2
# print(np.array([[2,1,0,0],
#                 [0,2,1,0],
#                 [0,0,2,1]]))

# 1.6.3
# print(mat.reshape(12,(1)))

# 1.6.4
# x_shape = tuple(map(int, input().split(' ')))
# X = np.fromiter(map(int, input().split(' ')), np.int).reshape(x_shape)
# y_shape = tuple(map(int, input().split(' ')))
# Y = np.fromiter(map(int, input().split(' ')), np.int).reshape(y_shape)
# print('matrix shapes do not match') if x_shape[1] != y_shape[1] else print(X.dot(Y.T))

# 1.6.5
# filename = input()
# f = urlopen(filename)
# load = np.loadtxt(f,dtype=float,skiprows=1,delimiter=',')
# print([np.mean(x) for x in zip(*load)])

# 1.7.3
# vec = [0+0.5, 1-0, 0-0.5, 0-1.5, 1-2.5, 2-3]
# print(max([sum(np.square(vec)), sum(np.abs(vec))]))

# 1.7.5
# x = np.array([[1,60],
#               [1,50],
#               [1,75]])
# y = np.array([[10],
#               [7],
#               [12]])
#
# s1 = x.T.dot(x)
# s2 = np.linalg.inv(s1)
# s3 = s2.dot(x.T)
# s4 = s3.dot(y)
# print(s4)

# 1.7.7
# fname = input()
# f = request.urlopen(fname)
# x = np.loadtxt(f, delimiter=',', skiprows=1)
# y = x[:,0:1].copy()
# x[:,0] = 1
# s1 = x.T.dot(x)
# s2 = np.linalg.inv(s1)
# s3 = s2.dot(x.T)
# s4 = s3.dot(y)
# print(' '.join([str(i) for i in s4.T[0]]))