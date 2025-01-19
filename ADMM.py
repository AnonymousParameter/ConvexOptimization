import numpy as np
from DataHandle import *

def soft_threshold(x, threshold):
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


def ADMM(A_all, b_all, lambda_reg, C, max_iter, tol):
    x_size = A_all.shape[1]
    xk = np.zeros(x_size)
    yk = np.zeros(x_size)
    vk = np.zeros(x_size)
    # print(xk)
    xk_list = []

    temp1 = np.linalg.inv(A_all.T @ A_all + C * np.eye(x_size, x_size))
    temp2 = A_all.T @ b_all
    for _ in range(max_iter):
        xk_list.append(xk)
        xk =  temp1 @ (temp2 - vk + C * yk)

        yk = soft_threshold(xk + vk/C, lambda_reg / C)
        vk += C* (xk - yk)

        if np.linalg.norm(xk - xk_list[-1], ord = 2) < tol:
            break
        print(np.linalg.norm(xk - xk_list[-1], ord = 2))
        
    # print(xk_list[-1])
    return xk_list
    # print(xk_list)
        # print(xk)
    # print(xk_list)

if __name__ == '__main__':
    lambda_reg = 0.01
    C = 1
    max_iter = 1000
    tol = 1e-5
    N = 10
    A_all, b_all, x_true = load_data(N)
    xk_list = ADMM(A_all, b_all, lambda_reg, C, max_iter, tol)
    dist_xk2true, dist_xk2opt = handle_data(xk_list, x_true)
    draw_line(dist_xk2true, dist_xk2opt, "proximal gradient with lambda = {}, C = {}, tol = {}".format(lambda_reg, C, tol))