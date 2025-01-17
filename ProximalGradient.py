def proximal_gradient(A_all, b_all, lambda_reg, x_true, x_opt, max_iter=1000, tol=1e-6):
    x_pgd = np.zeros(n)
    pgd_dist_true = []
    pgd_dist_opt = []
    
    # 计算Lipschitz常数 L = ||A_all||_2^2
    L = np.linalg.norm(A_all, ord=2)**2
    t = 1.0 / L  # 步长
    
    for iter in range(1, max_iter+1):
        grad = A_all.T @ (A_all @ x_pgd - b_all)
        x_temp = x_pgd - t * grad
        x_pgd = soft_threshold(x_temp, lambda_reg * t)
        
        # 记录距离
        dist_true = np.linalg.norm(x_pgd - x_true)
        dist_opt = np.linalg.norm(x_pgd - x_opt)
        pgd_dist_true.append(dist_true)
        pgd_dist_opt.append(dist_opt)
        
        # 检查终止条件
        if np.linalg.norm(A_all @ x_pgd - b_all) < tol:
            print(f'Proximal Gradient Method converged in {iter} iterations.')
            break
    else:
        print(f'Proximal Gradient Method reached maximum iterations ({max_iter}).')
    
    return pgd_dist_true, pgd_dist_opt