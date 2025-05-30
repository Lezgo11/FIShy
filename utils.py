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
        fill_col = int(ids[i+1])-int(ids[i])
        diag[fill_col, int(ids[i+n+1])] = filling_num
    
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