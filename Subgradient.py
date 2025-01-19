import numpy as np
from DataHandle import *
np.random.seed(0)
def subgradient_x(x, lambda_reg):
    sub_x = np.zeros(x.shape)
    for i, data in enumerate(x):
        if data == 0:
            sub_x[i] = np.random.uniform(-1, 1) * lambda_reg
        else:
            sub_x[i] = np.sign(x[i]) * lambda_reg
    
    return sub_x


def subgradient(A_all, b_all, lambda_reg, max_iter, tol):
    x_size = A_all.shape[1]
    xk = np.zeros(x_size)
    xk_list = []
    # print(xk)
    alpha = 0.0001
    alphak = alpha
    for i in range(max_iter):
        dk = (A_all.T @ (A_all @ xk - b_all)) + subgradient_x(xk, lambda_reg)
        xk = xk - alphak * dk
        alphak = alpha / (i+1)
        # print(xk)
        if xk_list != [] and np.linalg.norm(xk - xk_list[-1], ord = 2) < tol:
            # print(np.linalg.norm(xk - xk_list[-1], ord = 2))
            break
        # if xk_list != []:
            # print(np.linalg.norm(xk - xk_list[-1], ord = 2))
        xk_list.append(xk)

    return xk_list
    # print(xk_list)
        # print(xk)
    # print(xk_list)

if __name__ == '__main__':
    lambda_reg = 0.01
    max_iter = 2000
    # alpha = 0.01
    tol = 1e-5
    N = 10
    A_all, b_all, x_true = load_data(N)
    xk_list = subgradient(A_all, b_all, lambda_reg, max_iter, tol)
    dist_xk2true, dist_xk2opt = handle_data(xk_list, x_true)
    draw_line(dist_xk2true, dist_xk2opt, "proximal gradient with lambda = {}, alpha = 1/(k+1), tol = {}".format(lambda_reg, tol))