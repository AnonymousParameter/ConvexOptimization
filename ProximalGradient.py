import numpy as np
from DataHandle import *

def soft_threshold(x, threshold):
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


def proximal_gradient(A_all, b_all, lambda_reg, alpha, max_iter, tol):
    xk = np.zeros(A_all.shape[1])
    # print(xk)
    xk_list = [0, ]
    for _ in range(max_iter):
        xk_half = xk - alpha * np.dot(A_all.T, (A_all @ xk - b_all))
        # print(xk_half)
        xk = soft_threshold(xk_half, lambda_reg * alpha)
        if np.linalg.norm(xk - xk_list[-1], ord = 2) < tol:
            break
        xk_list.append(xk)
    xk_list.pop(0)
    # print(xk_list)
        # print(xk)
    return xk_list

 
# 显示图形
if __name__ == '__main__':
    lambda_reg = 10
    alpha = 0.001
    max_iter = 2000
    tol = 1e-6
    N = 10
    A_all, b_all, x_true = load_data(N)
    xk_list = proximal_gradient(A_all, b_all, lambda_reg, alpha, max_iter, tol)
    dist_xk2true, dist_xk2opt = handle_data(xk_list, x_true)
    draw_line(dist_xk2true, dist_xk2opt, "proximal gradient with lambda = {}, alpha = {}, tol = {}".format(lambda_reg, alpha, tol))