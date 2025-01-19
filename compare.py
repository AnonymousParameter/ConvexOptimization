import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os

ASize = (5, 200)
BSize = 5
XSize = 200

alpha = 0.005
lam = 10

def load_data():
    A=[]                
    b=[]
    X=[]
    data_folder = "python\ConvexOptimization\\randomData"
    # 使用循环加载每个文件并将数组添加到列表中
    for i in range(1, 11):
        Ai_filename= os.path.join(data_folder, f'A_{i}.npy')
        bi_filename= os.path.join(data_folder, f'b_{i}.npy')

        Ai = np.load(Ai_filename)
        bi = np.load(bi_filename)
        A.append(Ai)
        b.append(bi)
    x_filename= os.path.join(data_folder, f'x.npy')
    X = np.load(x_filename)
    return A,b,X

def thread_hold(lam,alpha,xk_half):
    xk_new=np.zeros(XSize)
    for i in range(XSize):
        if xk_half[i]>lam * alpha:
            xk_new[i]= xk_half[i] - lam * alpha
        elif xk_half[i]<-lam * alpha:
            xk_new[i]= xk_half[i] + lam * alpha
        else:
            xk_new[i]=0
    return xk_new

def proximal_gradient(A,b,X,alpha,lam,max_iter=2000):
    xk=np.zeros(XSize)
    dist2x_true=[]
    xk_sequence=[]
    delta=np.zeros(XSize)
    # for i in range(500):
    k=0
    while k<max_iter:
        delta=np.zeros(XSize)
        for j in range(10):
            # xk_half=xk-alpha*np.dot(A[j].T,np.dot(A[j],xk)-b[j])
            delta+=np.dot(A[j].T,np.dot(A[j],xk)-b[j])
        xk_half=xk-alpha*delta
        xk_new=thread_hold(lam,alpha,xk_half)
        if np.linalg.norm(xk_new-xk,ord=2)<1e-5:
            break
        dist2x_true.append(np.linalg.norm(xk_new-X,ord=2))
        xk_sequence.append(xk_new)
        xk=xk_new.copy()
        k+=1
    return xk,dist2x_true,xk_sequence


if __name__ == '__main__':
    A,b,X=load_data()
    xk,dist2x_true,xk_sequence=proximal_gradient(A,b,X,alpha,lam)
    dist2x_opt=[]
    for i, data in enumerate(xk_sequence):
        dist2x_opt.append(np.linalg.norm(data - xk, ord=2))

    plt.figure(num="LJW 21307381",figsize=(10, 5))
    # 创建子图布局
    gs = gridspec.GridSpec(2, 3,width_ratios=[1.5, 1, 1.5])

    # 创建主图，跨越2列3行
    ax_main = plt.subplot(gs[:, :2])
    plt.title("Combined Plot,lambda={}".format(lam))
    plt.plot(dist2x_true, label='X-true-distance')
    plt.plot(dist2x_opt, label='X-opt-distance')
    plt.legend()

    # 创建第一个副图
    ax1 = plt.subplot(gs[0, 2])
    plt.title("X-opt-distance")
    plt.plot(dist2x_opt, label='X-opt-distance')
    plt.legend()

    # 创建第二个副图
    ax2 = plt.subplot(gs[1, 2])
    plt.title("X-true-distance")
    plt.plot(dist2x_true, label='X-true-distance')
    plt.legend()

    
    # 调整布局
    plt.tight_layout()

    # 显示图形
    plt.show()