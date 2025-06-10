import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from timeit import default_timer as timer

def read_file (name):
    file = open(name,"r")
    lines = file.readlines()
    dimensions = lines[1].lstrip().rstrip().split(None,1)
    dim_matrix = int(dimensions[0])
    dim_arrays = int(dimensions[1])
    jm_read = np.zeros(dim_arrays).astype(int)
    vm_read = np.zeros(dim_arrays)
    for i in range (dim_arrays):
        aux = lines[i+2].lstrip().rstrip().split(None,1)
        jm_read[i] = int(aux[0])
        vm_read[i] = float(aux[1])
    file.close()
    return(jm_read,vm_read,dim_matrix)

def CSR_product(ia, j_m, v, x):
    ia_csr = ia.copy()
    j_m_csr = j_m.copy()
    v_csr = v.copy()
    x_csr = x.copy()
    y = np.zeros(len(x_csr))
    for i in range(len(ia_csr)-1):
        v_subarray = v_csr[ia_csr[i]:ia_csr[i+1]]
        j_subarray = j_m_csr[ia_csr[i]:ia_csr[i+1]]

        for j in range(len(v_subarray)):
            y[i] += x[j_subarray[j]]*v_subarray[j]
    
    return(y)

def CSC_product(ia, j_m, v, x):
    ia_csc = ia.copy()
    j_m_csc = j_m.copy()
    v_csc = v.copy()
    x_csc = x.copy()
    y = np.zeros(len(x_csc))
    for i in range(len(ia_csc)-1):
        j_m_subarray = j_m_csc[ia_csc[i]:ia_csc[i+1]]
        v_subarray = v_csc[ia_csc[i]:ia_csc[i+1]]
        v_subarray_mul = v_subarray*x_csc[i]
        for j in range (len(v_subarray)):
            row = j_m_subarray[j]
            y[row] += v_subarray_mul[j]

    return(y)

def MSR_product(jm, vm, x, bool):
    #jm -= 1- Let CSR do this index alignment.
    jm_msr = jm.copy()
    vm_msr = vm.copy()
    x_msr = x.copy()
    diagonal = vm_msr[0: len(x_msr)]
    v_subarray = vm_msr[len(x_msr)+1: len(vm_msr)]
    ia_array = jm_msr[0:len(x_msr)+1]
    ia_array -= len(x_msr)+1
    j_array = jm[len(x_msr)+1:len(jm_msr)]
    if bool == False:
        y_offdiagonal = CSR_product(ia_array, j_array, v_subarray, x)
    else:
        y_offdiagonal = CSR_product(ia_array, j_array, v_subarray, x) + CSC_product(ia_array, j_array, v_subarray, x)

    y_diagonal = diagonal*x_msr
    return(y_offdiagonal+y_diagonal)

def backward_subs(m_function, b_vector):
    matrix = m_function.copy()
    x_sol = np.zeros(len(b_vector))
    m = len(b_vector)
    sum = 0
    for i in range (m):
        sum += b_vector[-1-i]
        for j in range(i):
            sum -= x_sol[-1-j]*matrix[-1-i,-1-j]
        if matrix[-1-i,-1-i] == 0:
            print("Matrix is singular")
        else:
            x_sol[-1-i] = sum/matrix[-1-i,-1-i]
            sum = 0
    return(x_sol)

def foward_subs(matrix , b):
    n = len(b)
    x_sol = np.zeros(n)
    sum = 0
    for i in range (n):
        sum += b[i]
        for j in range (i):
            sum -= matrix[i,j]*x_sol[j]
        if matrix[i,i] == 0:
            print("Matrix is singular")
        else:
            x_sol[i]= sum/matrix[i,i]
        sum = 0
    return(x_sol)

def lower_triangular(jm, vm, size):
    matrix_L =np.zeros((size, size))
    x_j = np.zeros(size)
    for j in range (size):
        x_j[j] = 1
        column_j = MSR_product(jm, vm, x_j, False)
        for i in range (j):
            column_j[i] = 0
        matrix_L[:,j] = column_j
        x_j[j] = 0
    return(matrix_L)

def precon_matrix(jm, vm, precon_method, size):
    if precon_method == "identity":
        m_matrix = 1.
    elif precon_method == "jacobi":
        m_matrix = vm[0:size]
    elif precon_method == "gauss":
        m_matrix = lower_triangular(jm, vm, size)
    return(m_matrix)

def precon_lin_sys(matrix_m, b_precon_sys, precon_method):
    size = len(b_precon_sys)
    x_sol = np.zeros(size)
    if precon_method == "identity":
        x_sol = b_precon_sys
    elif precon_method == "jacobi":
        matrix_func = matrix_m.copy()
        x_sol = np.multiply(np.reciprocal(matrix_func), b_precon_sys)
    elif precon_method == "gauss":
        matrix_func = matrix_m.copy()
        x_sol = foward_subs(matrix_func , b_precon_sys)
    return(x_sol)

def arnoldi_alg (jm, vm, v_0, m, matrix_m, precon_method):
    hessenberg_m = np.zeros((m+1,m))
    orthonormal_m = np.zeros((len(v_0),m))
    v_next = v_0 / LA.norm(v_0)
    for j in range(m):
        orthonormal_m[:,j] = v_next
        v_next = MSR_product(jm, vm, orthonormal_m[:,j], False)
        v_next = precon_lin_sys(matrix_m , v_next, precon_method)
        for i in range (j+1):
            hessenberg_m[i,j] = np.dot(v_next, orthonormal_m[:,i])
            v_next -= hessenberg_m[i,j]*orthonormal_m[:,i]
        dot_product_max = 0
        for s in range (j):
            dot_product_j = np.dot(v_next,orthonormal_m[:,s])
            if dot_product_j > dot_product_max:
                dot_product_max = dot_product_j
            
        hessenberg_m[j+1,j] = LA.norm(v_next)
        if LA.norm(v_next) == 0:
            print("Norm equal to zero")
        else:
            v_next = v_next/LA.norm(v_next)
    return(orthonormal_m, hessenberg_m, dot_product_max)

def givens_rotation(hessenber_m_func , norm_v_o):
    H_m = hessenber_m_func.copy()
    m = len(H_m[:,0])-1
    g = np.zeros(m+1)
    g[0] = norm_v_o
    for r in range(m):
        h = (H_m[r,r]**2 +H_m[r+1,r]**2)**(0.5)
        c = H_m[r,r]/h
        s = H_m[r+1,r]/h
        rotation_m = np.array([[c, s], [-s, c]])
        g[r:r+2] = np.matmul(rotation_m,g[r:r+2])
        H_m[r,r] = h
        H_m[r+1,r] = 0
        for j in range (r+1,m):
            H_m[r:r+2,j] = np.matmul(rotation_m,H_m[r:r+2,j])
    return(H_m , g)

def gmres(j_m, v_m, x_0, b , relative_residual, max_iterations, precon_method):    
    r_0 = b - MSR_product(j_m, v_m, x_0, False)
    matrix_m = precon_matrix(j_m , v_m , precon_method, len(b))
    r_0_precon = precon_lin_sys(matrix_m, r_0, precon_method)
    norm_r_0_precon = np.linalg.norm(r_0_precon)
    relative_residual_iteration = 10**4
    dimension_krylov = 0
    r_residual_list = []
    max_inner_product_list = []
    while (relative_residual_iteration> relative_residual) and (dimension_krylov < max_iterations):
        dimension_krylov += 10
        orthonormal_base, hessenberg, max_dot_product = arnoldi_alg(j_m , v_m , r_0_precon , dimension_krylov, matrix_m , precon_method)
        max_inner_product_list.append(max_dot_product)
        upper_triangular, g_vector = givens_rotation(hessenberg , norm_r_0_precon)
        upper_triangular = upper_triangular[:-1]
        residual = abs(g_vector[-1])
        relative_residual_iteration = residual/norm_r_0_precon
        r_residual_list.append(relative_residual_iteration)
    g_vector_r = g_vector[:-1]
    proj_b_krylov = backward_subs(upper_triangular, g_vector_r)
    x_best_estimate = x_0 + np.matmul(orthonormal_base , proj_b_krylov)
    return(x_best_estimate, r_residual_list, max_inner_product_list)

def CG( j_m_matrix, v_m_matrix,  x_0, b, iterations_max, x_sol, relative_residual_goal):
    x = x_0.copy()
    relative_residual =10**3
    counter = 0
    norm_A_error_list = []
    norm_residual_list = []
    r = b - MSR_product(j_m_matrix , v_m_matrix , x, True)
    norm_r_0 = LA.norm(r)
    p = r.copy()
    alpha = 0
    beta = 0
    while (relative_residual> relative_residual_goal) and (counter < iterations_max):
        error = x_sol-x
        image_error = MSR_product(j_m_matrix , v_m_matrix , error, True)
        matrix_vector = MSR_product(j_m_matrix , v_m_matrix , p, True)
        norm_A_error = np.dot(image_error, error)**0.5
        norm_A_error_list.append(norm_A_error)
        dot_product_m = np.dot(matrix_vector,p)
        dot_product_r = np.dot(r,r)
        norm_residual_list.append(dot_product_r**0.5)
        alpha = np.dot(r,r)/dot_product_m
        x += alpha*p
        r -= alpha*matrix_vector
        beta = np.dot(r,r)/dot_product_r
        p = r +beta*p
        relative_residual = LA.norm(r)/norm_r_0
        counter+=1
    return(x, norm_residual_list, norm_A_error_list)

# Call main function
def main_gmres():
    #Name of the files
    file_gmres = 'Data_P1/gmres_matrix_msr_1.txt'
    #file_cg = "cg_test_msr.txt"
    j_m, v_m, dim_matrix = read_file(file_gmres)
    j_m -= 1
    x_0 = np.zeros(dim_matrix)
    x_sol = np.ones(dim_matrix)
    b_image = MSR_product(j_m, v_m, x_sol, False)
    relative_residual_goal = 10**-8
    restarted_gmres = 600
    precon_method = "identity"
    #start_time = timer()
    x_iterative_sol, r_residual_list, max_inner_product = gmres(j_m, v_m, x_0, b_image, relative_residual_goal, restarted_gmres, precon_method)
    x_axis = np.arange(10,10*(len(r_residual_list)+1), 10)
    print(f"Error eucledean norm: {LA.norm(x_sol - x_iterative_sol)}")
    #print(len(r_residual_list))
    #Plots Full GMRES
    plt.semilogy(x_axis,r_residual_list)
    #plt.semilogy(x_axis , max_inner_product)
    plt.xlabel("Number of iterations")
    plt.ylabel("Relative residual")
    plt.title(f"Preconditioned GMRES {precon_method} Relative Residual")
    plt.show()

def main_cg():
    file_cg = "cg_test_msr.txt"
    j_m, v_m, dim_matrix = read_file(file_cg)
    j_m = j_m-1
    #dim_matrix = 6
    x_sol = np.ones(dim_matrix)
    x_ini = np.zeros(dim_matrix)
    b_image = MSR_product(j_m, v_m, x_sol, True)
    relative_residual_goal = 10**-6
    num_iterations_max = 10**4
    x_solution, norm_residual_list, norm_A_error_list = CG(j_m , v_m, x_ini, b_image, num_iterations_max, x_sol, relative_residual_goal)
    x_axis = np.arange(0,len(norm_A_error_list))
    print(f"Error eucledean norm: {LA.norm(x_sol-x_solution)}")
    plt.semilogy(x_axis, norm_residual_list)
    plt.xlabel("Number of iterations")
    plt.ylabel("Relative Residual")
    plt.title("CG Method norm of the residual")
    plt.show()
    plt.semilogy(x_axis, norm_A_error_list)
    plt.xlabel("Number of iterations")
    plt.ylabel("Norm A of the residual")
    plt.title("CG Method A norm of the error")
    plt.show()

main_gmres()
#main_cg()

#y_sol = CSR_product(m_ia-1,j_array-1, v_array, x_array)
#y_sol = CSC_product(m_ia-1, j_array-1, v_array, x_array)
#y_sol = MSR_product(j_m-1, v_m, x)
#print(y_sol)