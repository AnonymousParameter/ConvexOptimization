import numpy as np
import os
import matplotlib.pyplot as plt

def load_data(N):
    current_file_path = os.path.realpath(__file__)
    current_folder_path = os.path.dirname(current_file_path)
    # print(current_folder_path)
    data_folder = os.path.join(current_folder_path, 'randomData')
    if not os.path.exists(data_folder):
        raise ValueError("数据不存在")
    
    A = []
    b = []

    for i in range(N):
        A_i_path = os.path.join(data_folder, 'A_{}.npy'.format(i+1))
        b_i_path = os.path.join(data_folder, 'b_{}.npy'.format(i+1))

        Ai = np.load(A_i_path)
        bi = np.load(b_i_path)
        A.append(Ai)
        b.append(bi)
    
    A_all = np.vstack(A)
    b_all = np.hstack(b)
    x_path = os.path.join(data_folder, 'x.npy'.format(i+1))
    x = np.load(x_path)

    return A_all, b_all, x

def handle_data(xk_list, x_true):
    dist_xk2true = []
    dist_xk2opt = []
    x_opt = xk_list[-1]
    for xk in xk_list:
        dist_xk2true.append(np.linalg.norm(xk - x_true, 2))
        dist_xk2opt.append(np.linalg.norm(xk - x_opt, 2))
    return dist_xk2true, dist_xk2opt

def draw_line(dist_xk2true, dist_xk2opt, name):
    x = list(range(len(dist_xk2true)))

    plt.plot(x, dist_xk2true, "r-", label="Distance between xk and true value")
    plt.plot(x, dist_xk2opt, "b-", label="Distance between xk and optimal value")


    # 标题设置
    plt.title(name)
    plt.xlabel("interation number")
    plt.ylabel("distance")
    # 图例设置
    plt.legend()
    plt.show()

if __name__ == '__main__':
    print(load_data(10))

