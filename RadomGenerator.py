import numpy as np
import os

def RandomDataGenerator():
    current_file_path = os.path.realpath(__file__)
    current_folder_path = os.path.dirname(current_file_path)
    # print(current_folder_path)
    data_folder = os.path.join(current_folder_path, 'randomData')
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    num_nodes = 10
    m = 5
    n = 200
    k = 5

    x = np.zeros(n)
    x_index = np.random.choice(n, k, replace = False)
    x[x_index] = np.random.randn(k)

    for i in range(num_nodes):
        A = np.random.normal(0, 1, (m, n))
        e = np.random.normal(0, 0.1, m)
        b = np.dot(A, x) + e

        np.save(os.path.join(data_folder, 'A_{}.npy'.format(i+1)), A)
        np.save(os.path.join(data_folder, 'b_{}.npy'.format(i+1)), b)

    np.save(os.path.join(data_folder, "x.npy"), x)

if __name__ == '__main__':
    RandomDataGenerator()