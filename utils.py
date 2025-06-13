import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spilu

#---------- MSR decomp.
def msr_to_matrix(filename):
    # Read data from the file
    with open(filename, 'r') as f:
        # Read the first line as a string
        SorN = [next(f)]
        lines = f.readlines()
    SorN= SorN[0].strip().split()
    header = lines[0].strip().split()
    n = int(header[0])
    fillings = []
    data = [line.strip().split() for line in lines[1:]]
    ids = np.array([int(item[0]) for item in data])
    ids_rows = ids[0:n+1]  # Extracting row indices
    ids_cols = ids[n+1:len(ids)]  # Extract column indices
    ids_cols = ids_cols-1  # Adjust column indices to be zero-based
    #print("Column indices:", ids_cols)

    for i in range(len(ids_rows) - 1): # only consider the row elements
        fillings.append(ids_rows[i+1] - ids_rows[i])  # append filling numbers

    #print("amoun of nonzero values per row:", fillings)

    vals = np.array([float(item[1]) for item in data])  # Convert to floats and extract values of matrix

    diag = np.diag(vals[0:n]) # Read diagonal elements
    vals = vals[n+1:len(vals)]  # Exclude diagonal elements from vals
    #print(f"values: {vals}")

    matrix = np.zeros((n, n)) # Initialize the matrix



    dict_fillings = {}  # Initialize dict_fillings as an empty dictionary
    for i in range(len(fillings)):
        dict_fillings[i] = fillings[i]

    # Use dictionaries to fill the matrix
    for i in range(n):
        if dict_fillings[i] == 0:
            continue  # Skip rows with no fillings
        start_idx = sum(fillings[:i])  # Calculate the starting index for the current row
        end_idx = start_idx + fillings[i]  # Calculate the ending index for the current row
        for idx, j in enumerate(ids_cols[start_idx:end_idx]):
            if idx < len(vals):  # Ensure we don't exceed the length of vals
                matrix[i, j] = vals[start_idx + idx]
    if SorN[0] == "s":
        matrix = matrix + matrix.T  # Make the matrix symmetric

    matrix = matrix + diag

    return matrix
#---------- GMRES

# 1. Make system of equations Ax=b
def make_system(decompressed_matrix):
    """Create a system of equations Ax = b with a known solution x."""
    n = decompressed_matrix.shape[0]
    x_true = np.ones(n)
    b = decompressed_matrix @ x_true
    x0 = np.zeros(n)
    return decompressed_matrix, b, x_true,x0 

# 2. Define Arnoldi method (GetKrylov)

def GetKrylovw(A, v0, k):
    """
    Arnoldi algorithm (Krylov approximation of a matrix)
    
    Inputs:
        A: Square matrix (n x n)
        v0: Initial vector (n,) or (n, 1)
        k: Number of Krylov steps
        
    Returns:
        V: Matrix (n x k) containing orthonormal basis of Krylov subspace
        H: Hessenberg matrix (k+1 x k)
    """
    v0 = v0.reshape(-1)  # Ensure v0 is a 1D array
    n = len(v0)
    #inputtype = A.dtype.type
    
    V = np.zeros((n, k + 1), dtype=v0.dtype)  # +1 for possible breakdown
    H = np.zeros((k + 1, k), dtype=v0.dtype)
    
    V[:, 0] = v0 / np.linalg.norm(v0)
    
    for m in range(k):
        vt = A @ V[:, m]  # Use @ instead of * for matrix-vector product
        for j in range(m + 1):
            H[j, m] = np.dot(V[:, j].conj(), vt)
            vt = vt - H[j, m] * V[:, j]
        H[m + 1, m] = np.linalg.norm(vt)
        if H[m + 1, m] < 1e-14:  # Handle breakdown
            break
        V[:, m + 1] = vt / H[m + 1, m]

    return V,H

# 3. TODO: Add GMRES 
def gmres():
    """
    Generalized Minimal Residual Method (GMRES) for solving linear systems.
    """

    # Load MSR and get full matrix
    A_msr = msr_to_matrix(msr_filename)
    A, b, x_true, x0 = make_system(A_msr)

    # Convert full matrix to MSR format
    j_m, v_m = msr_to_matrix(A)

    # Initial residual
    r0 = b - MSR_product(j_m, v_m, x0, False)

    # Preconditioning
    m_matrix = precon_matrix(j_m, v_m, precon_method, len(b))
    r0_precon = precon_lin_sys(m_matrix, r0, precon_method)
    norm_r0_precon = np.linalg.norm(r0_precon)

    relative_residual_iteration = 1e4
    dimension_krylov = 10
    r_residual_list = []
    max_inner_product_list = []

    while (relative_residual_iteration > relative_residual) and (dimension_krylov < max_iterations):
        # Expand Krylov dimension by 10
        dimension_krylov += 10

        # Arnoldi + preconditioning
        V, H, max_dot_product = arnoldi_alg(j_m, v_m, r0_precon, dimension_krylov, m_matrix, precon_method)
        max_inner_product_list.append(max_dot_product)

        # Givens rotations
        H_tri, g_vector = givens_rotation(H, norm_r0_precon)
        H_tri = H_tri[:-1]  # Remove last row for square system
        residual = abs(g_vector[-1])
        relative_residual_iteration = residual / norm_r0_precon
        r_residual_list.append(relative_residual_iteration)

    # Backward solve
    y = backward_subs(H_tri, g_vector[:-1])
    x = x0 + V @ y

    return x, r_residual_list, max_inner_product_list




# 4. TODO: Add GMRES with preconditioning
#---------- CG
# 5. Conjugate Gradient Method
def conjugate_gradient(A, b, x0, x_true, tol = 1e-8, max_iter=None):
    """
    Conjugate Gradient method without preconditioning.
    Also tracks residuals and A-norm error.
    """
    x = x0.copy()
    r = b - A @ x
    p = r.copy()
    # Initial residual and norms
    rsold = np.dot(r,r)
    r_norms = [np.sqrt(rsold)]
    e_A_norms = [np.sqrt((x - x_true).T @ A @ (x - x_true))]
    
    if max_iter is None:
        max_iter = len(b)

    for i in range(max_iter):
        Ap = A @ p
        alpha = rsold / np.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = np.dot(r, r)
        
        r_norms.append(np.sqrt(rsnew))
        e = x - x_true
        e_A_norm = np.sqrt(e.T @ A @ e)
        e_A_norms.append(e_A_norm)

        if np.sqrt(rsnew) / np.sqrt(r_norms[0]) < tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew

    return x, r_norms, e_A_norms
# 6. #TODO: add preconditioned CG

#---------- Other tests
def load_matrix(filename):
    """
    Load a matrix from a file in MSR format.
    Args:
        filename (str): Path to the file containing the matrix data
    Returns:
        numpy.ndarray: The loaded matrix
    """
    l = []
    with open(filename, "r") as f:
        next(f)
        for line in f:
            line = line.strip().split()  # Remove leading/trailing whitespace and split by spaces
            row = [float(x) for x in line]  # Convert each element to a float
            l.append(row)
    # Convert list to numpy array
    return matrix



#Givens rotations
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

#-
# Backward substitution
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


def ilu0(A):
    """
    Incomplete LU factorization (ILU(0)) for a dense square matrix A.
    Only keeps elements in L and U where A has non-zero entries.
    """
    A = A.copy()
    n = A.shape[0]
    L = np.eye(n)
    U = np.zeros_like(A)

    for i in range(n):
        for j in range(n):
            if A[i, j] == 0:
                continue
            if j < i:
                s = sum(L[i, k] * U[k, j] for k in range(j))
                if U[j, j] == 0:
                    raise ZeroDivisionError(f"Zero pivot encountered at U[{j},{j}]")
                L[i, j] = (A[i, j] - s) / U[j, j]
            else:
                s = sum(L[i, k] * U[k, j] for k in range(i))
                U[i, j] = A[i, j] - s

    return L, U


def gmres_2(msr_filename, relative_residual=1e-8, k_dim=10, max_iterations=600, precon_method="identity"):
    A_msr = msr_to_matrix(msr_filename)
    A, b, x_true, x0 = make_system(A_msr)

    # ILU preconditioner object
    ilu_solver = None
    # Preconditioner matrix generator
    def precon_matrix(A, precon_method):
        if precon_method == "identity":
            return np.eye(A.shape[0])
        elif precon_method == "jacobi":
            return np.diag(1.0 / np.diag(A))
        elif precon_method == "gauss":
            L = np.tril(A)
            return np.linalg.inv(L)
        elif precon_method == "ilu":
            L, U = ilu0(A)
            return (L,U)
        else:
            raise ValueError("Unknown preconditioner")

    # Apply preconditioner (as a matrix-vector product)
    def apply_preconditioner(M, r):
        if precon_method == "ilu":
            L, U = M
            y = forward_substitution(L, r)
            return backward_substitution(U, y)
        else:
            return  M @ r

    def forward_substitution(L, b):
        n = len(b)
        y = np.zeros_like(b)
        for i in range(n):
            y[i] = b[i] - np.dot(L[i, :i], y[:i])
        return y
    
    def backward_substitution(U, y):
        n = len(y)
        x = np.zeros_like(y)
        for i in reversed(range(n)):
            if U[i, i] == 0:
                raise ZeroDivisionError(f"Zero pivot encountered at U[{i},{i}]")
            x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
        return x

    # Modified Arnoldi process with preconditioning
    def arnoldi_with_precon(A, M, r0_precon, k):
        n = len(r0_precon)
        V = np.zeros((n, k+1))
        H = np.zeros((k+1, k))
        max_inner = 0.0

        beta = np.linalg.norm(r0_precon)
        V[:, 0] = r0_precon / beta

        for j in range(k):
            z = apply_preconditioner(M, A @ V[:, j])  # Apply preconditioner to Av_j

            for i in range(j + 1):
                H[i, j] = np.dot(V[:, i], z)
                z = z - H[i, j] * V[:, i]
                max_inner = max(max_inner, abs(H[i, j]))

            H[j+1, j] = np.linalg.norm(z)
            if H[j+1, j] != 0 and j + 1 < k:
                V[:, j+1] = z / H[j+1, j]

        return V[:, :k], H[:k+1, :k], max_inner

    # Givens rotation QR method
    def givens_rotation(H, beta):
        m, n = H.shape
        R = np.copy(H)
        cs = np.zeros(n)
        sn = np.zeros(n)
        g = np.zeros(m)
        g[0] = beta

        for i in range(n):
            r = np.hypot(R[i, i], R[i+1, i])
            cs[i] = R[i, i] / r
            sn[i] = R[i+1, i] / r
            R[i, i] = r
            R[i+1, i] = 0.0

            g[i+1] = -sn[i] * g[i]
            g[i] = cs[i] * g[i]

            for j in range(i+1, n):
                temp = cs[i] * R[i, j] + sn[i] * R[i+1, j]
                R[i+1, j] = -sn[i] * R[i, j] + cs[i] * R[i+1, j]
                R[i, j] = temp

        return R[:n, :], g[:n+1]

    # Back substitution for upper-triangular systems
    def backward_subs(R, g):
        n = R.shape[1]
        x = np.zeros(n)
        for i in reversed(range(n)):
            x[i] = (g[i] - np.dot(R[i, i+1:], x[i+1:])) / R[i, i]
        return x

    # Begin GMRES loop
    r0 = b - A @ x0
    M = precon_matrix(A, precon_method)
    r0_precon = apply_preconditioner(M, r0)
    norm_r0_precon = np.linalg.norm(r0_precon)

    rel_residual = 1e4
    residuals = []
    inner_products = []
    starting_k = 1

    while rel_residual > relative_residual and starting_k < max_iterations:
        V, H, max_ip = arnoldi_with_precon(A, M, r0_precon, starting_k)
        inner_products.append(max_ip)

        R, g = givens_rotation(H, norm_r0_precon)
        rel_residual = abs(g[-1]) / norm_r0_precon
        residuals.append(rel_residual)
        starting_k += k_dim

    y = backward_subs(R, g[:-1])
    x = x0 + V @ y

    x_axis = np.arange(k_dim,k_dim*(len(residuals)+1), k_dim)

    return x, residuals, x_axis


