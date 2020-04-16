import numpy as np
from scipy import misc
import imageio
import matplotlib.pyplot as plt
import os
from contrast_measurer import image_contrast
import json
 

def julia(output, m = 480, n = 320, s = 300, a = -0.4, b = 0.5, grayscale = False):
    x = np.linspace(-m / s, m / s, num=m).reshape((1, m))
    y = np.linspace(-n / s, n / s, num=n).reshape((n, 1))
    Z = np.tile(x, (n, 1)) + 1j * np.tile(y, (1, m))
    
    C = np.full((n, m), complex(a, b))
    M = np.full((n, m), True, dtype=bool)
    N = np.zeros((n, m))
    for i in range(256):
        Z[M] = Z[M] * Z[M] + C[M]
        M[np.abs(Z) > 2] = False
        N[M] = i
    
    if grayscale:
        # imageio.imwrite(output, np.flipud(1 - M))
        # This is the grayscale version.
        imageio.imwrite(output, np.flipud(255 - N))
        
    
    else:
        # Save with Matplotlib using a colormap.
        # This is the color version
        fig = plt.figure()
        fig.set_size_inches(m / 100, n / 100)
        ax = fig.add_axes([0, 0, 1, 1], frameon=False, aspect=1)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(np.flipud(N))
        plt.savefig(f'.{output}')
        plt.close()


ans = {
    'HDR': [],
    'SDR': []
}

file_count = 0
directories = [item for item in os.listdir() if '.' not in item and '__' not in item]

for directory in directories:
    files = os.listdir(f'./{directory}')
    for f in files:  
        if 'Fractal' not in f:
            file_count += 1


for directory in directories:
    files = os.listdir(f'./{directory}')
    for f in files:  
        if 'Fractal' not in f:
            mean, var = image_contrast(f'./{directory}/{f}')
            ans['HDR' if 'HDR' in f else 'SDR'].append((mean, var))
            file_count -= 1
            print(f'Analyzed {f}. {file_count} files left...')
        


json.dump(ans, open("contrast_data.json", 'w'), indent=4)
