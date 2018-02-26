import numpy as np

def linear_size(output):
    output_size = np.array(output.size())
    h, w = output_size[2], output_size[3]
    size = int(h * w)
    return size