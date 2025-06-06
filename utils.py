import numpy as np
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
    inputtype = A.dtype.type
    
    n = A.shape[0]
    V = np.zeros((n, k + 1), dtype=inputtype)  # +1 for possible breakdown
    H = np.zeros((k + 1, k), dtype=inputtype)
    
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

#Stinky krylov attempt below!!!!!!!
def GetKrylov(A, v0, k):
    """
    Arnoldi algorithm to compute a Krylov approximation of matrix A.
    
    Parameters:
        A : ndarray
            Square matrix of shape (n, n)
        v0 : ndarray
            Initial vector of shape (n,) or (n, 1)
        k : int
            Number of Krylov steps
            
    Returns:
        V : ndarray
            Matrix of orthonormal vectors of shape (n, k+1)
        H : ndarray
            Upper Hessenberg matrix of shape (k+1, k)
    """
    v0 = v0.reshape(-1, 1)
    V = v0 / np.linalg.norm(v0)
    V = V.astype(A.dtype)
    V_list = [V]
    H = np.zeros((k + 1, k), dtype=A.dtype)

    for m in range(k):
        w = A @ V_list[m]
        for j in range(m + 1):
            H[j, m] = np.conj(V_list[j].T) @ w
            w = w - H[j, m] * V_list[j]
        H[m + 1, m] = np.linalg.norm(w)
        if H[m + 1, m] < 1e-14:
            break  # early termination
        V_list.append(w / H[m + 1, m])
        
    V = np.hstack(V_list)
    return V, H


# 3. TODO: Add GMRES 
def gmres():
    """
    Generalized Minimal Residual Method (GMRES) for solving linear systems.
    """
    
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
