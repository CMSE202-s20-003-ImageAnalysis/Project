import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
 

def julia(m = 480, n = 320, s = 300, c = -0.4 + 0.6j):
    x = np.linspace(-m / s, m / s, num=m).reshape((1, m))
    y = np.linspace(-n / s, n / s, num=n).reshape((n, 1))
    Z = np.tile(x, (n, 1)) + 1j * np.tile(y, (1, m))
    
    C = np.full((n, m), c)
    M = np.full((n, m), True, dtype=bool)
    N = np.zeros((n, m))
    for i in range(256):
        Z[M] = Z[M] * Z[M] + C[M]
        M[np.abs(Z) > 2] = False
        N[M] = i
    
    # misc.imsave('julia-m.png', np.flipud(1 - M))
    # misc.imsave('julia.png', np.flipud(255 - N))
    
    # Save with Matplotlib using a colormap.
    fig = plt.figure()
    fig.set_size_inches(m / 100, n / 100)
    ax = fig.add_axes([0, 0, 1, 1], frameon=False, aspect=1)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.imshow(np.flipud(N))
    plt.savefig('julia-plt.png')
    plt.close()

julia(c = 0.34939021533559678 + 0.1164246053613936j)