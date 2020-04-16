import os
import shutil
import numpy as np

dirs = [name for name in os.listdir() if '.' not in name and '__' not in name]
for directory in dirs:
    files = os.listdir(directory)
    for image_name in files:
        if 'Fractal' in image_name:
            source = "./{}/{}".format(directory, image_name)
            
            destination = "./data__dir_fractal/{}/{}/{}".format('train' if np.random.random() < 0.75 else 'val','HDR' if 'HDR' in image_name else 'SDR', f'{directory}-{image_name}')
            
            # Copy the content of 
            # source to destination 
            dest = shutil.copyfile(source, destination) 