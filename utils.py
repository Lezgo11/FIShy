import numpy as np
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

def conjugate_gradient(A, b, x0, x_true, tol=1e-8, max_iter=None):
    """
    Conjugate Gradient method without preconditioning.
    Also tracks residuals and A-norm error.
    """
    x = x0.copy()
    r = b - A @ x
    p = r.copy()
    rsold = np.dot(r, r)
    
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