import os
import shutil
import numpy as np

dirs = [name for name in os.listdir() if '.' not in name and '__' not in name]
for directory in dirs:
    files = os.listdir(directory)
    for image_name in files:
        if 'Fractal' in image_name:
            source = "./{}/{}".format(directory, image_name)
            
            destination = "./data__dir_binary_fractal/{}/{}".format('positive' if 'HDR' in image_name else 'negative', f'{directory}-{image_name}')
            
            # Copy the content of 
            # source to destination 
            dest = shutil.copyfile(source, destination) 