import numpy as np
#---------- MSR decomp.
def decompress_msr(filename):
    """
    Decompress a matrix stored in Modified Compressed Sparse Row (MSR) format from a text file.
    Args:
        filename (str): Path to the text file containing the MSR format data     
    Returns:
        numpy.ndarray: The decompressed matrix
    """

    # Read data from the file
    with open(filename, 'r') as f:
        next(f)
        lines = f.readlines()
    
    
    header = lines[0].strip().split()
    n = int(header[0])

    data = [line.strip().split() for line in lines[1:]]
    ids = np.array([int(item[0]) for item in data])  # Convert to integers
    vals = np.array([float(item[1]) for item in data])  # Convert to floats

    # Initialize the matrix
    diag = np.diag(vals[0:n]) # Read diagonal elements
    # Read off-diagonal elements
    for i in range(3):
        filling_num = vals[i+n+1]
        fill_col = ids[i+1]-ids[i]
        diag[fill_col, ids[i+n+1]] = filling_num
    
    return diag

def msr_to_dense(filename):
    # Read data from the file
    with open(filename, 'r') as f:
        next(f)
        lines = f.readlines()

    lines = lines[0].strip().split('\n')
    symmetry = lines[0].strip()
    n, nnz = map(int, lines[1].split())
    
    # Parse the MSR arrays
    val = []
    bindx = []
    for line in lines[2:2+n+1]:  # First n+1 entries are diagonal + marker
        parts = line.split()
        bindx.append(int(parts[0]))
        if len(parts) > 1:
            val.append(float(parts[1]))
    
    # Remaining entries are off-diagonal values and their column indices
    for line in lines[2+n+1:2+nnz]:
        parts = line.split()
        bindx.append(int(parts[0]))
        if len(parts) > 1:
            val.append(float(parts[1]))
    
    # Initialize dense matrix with zeros
    dense = [[0.0 for _ in range(n)] for _ in range(n)]
    
    # Fill diagonal elements
    for i in range(n):
        dense[i][i] = val[i]
    
    # Process off-diagonal elements
    off_diag_start = bindx[0]  # First entry in bindx points to where off-diag values start
    off_diag_val = val[n+1:]   # Off-diag values start after diagonal + marker
    
    # The bindx entries before off_diag_start give row pointers
    row_ptr = bindx[:n+1]
    
    for i in range(n):
        start = row_ptr[i] - off_diag_start
        end = row_ptr[i+1] - off_diag_start if i+1 < len(row_ptr) else len(off_diag_val)
        
        for idx in range(start, end):
            col = bindx[off_diag_start + idx] - 1  # Convert to 0-based index
            value = off_diag_val[idx]
            dense[i][col] = value
            if symmetry == 's' and i != col:  # Mirror for symmetric matrices
                dense[col][i] = value
    
    return dense
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
def GetKrylov(A, v0, k):
    """
    Arnoldi algorithm (Krylov approximation of a matrix)
        input: 
            A: matrix to approximate
            v0: initial vector (should be in matrix form) 
            k: number of Krylov steps 
        output: 
            V: matrix (large, N*k) containing the orthogonal vectors
            H: matrix (small, k*k) containing the Krylov approximation of A
    """
    v0 = v0.reshape(-1, 1)  # Ensure x_true is a column vector
    #print 'ARNOLDI METHOD'
    inputtype = A.dtype.type
    V = np.matrix(v0.copy() / np.linalg.norm(v0), dtype=inputtype)
    H = np.matrix( np.zeros((k+1,k), dtype=inputtype) )
    for m in range(k):
        vt = A*V[ :, m]
        for j in range( m+1):
            H[ j, m] = (V[ :, j].H * vt )[0,0]
            vt -= H[ j, m] * V[:, j]
        H[ m+1, m] = np.linalg.norm(vt);
        if m is not k-1:
            V =  np.hstack( (V, vt.copy() / H[ m+1, m] ) ) 
    return V,  H

# 3. TODO: Add GMRES 
def gmres():
    """
    Generalized Minimal Residual Method (GMRES) for solving linear systems.
    """
    
# 4. TODO: Add GMRES with preconditioning
#---------- CG
# 5. Conjugate Gradient Method
def conjugate_gradient(A, b, x0, x_true, tol=1e-8, max_iter=None):
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
