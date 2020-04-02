from PIL import Image
import numpy as np

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

# Taken from https://stackoverflow.com/questions/9733288/how-to-programmatically-calculate-the-contrast-ratio-between-two-colors
# Translated into python by ourselves.
def luminanace(r, g, b):
    a = [r, g, b]
    for i in range(len(a)):
        color = a[i]
        color /= 255
        if color <= 0.03928:
            a[i] = color / 12.92
        else:
            a[i] = ((color + 0.055) / 1.055)**2.4
    
    return a[0] * 0.2126 + a[1] * 0.7152 + a[2] * 0.0722

def contrast(rgb1, rgb2): 
    result = (luminanace(rgb1[0], rgb1[1], rgb1[2]) + 0.05) / (luminanace(rgb2[0], rgb2[1], rgb2[2]) + 0.05)
    
    if result < 1: 
        result = 1/result
    
    return result

def calculate_average_contrast_with_other_pixels(im, pixel):
    contrasts = []
    for row in range(im.shape[0]): 
        for column in range(im.shape[1]):
            contrasts.append(contrast(pixel, im[row][column]))
    return np.mean(contrasts)


def image_contrast(path):
    im = np.asarray(Image.open(path).resize((30, 30)))
    avg_contrasts = []
    for row in range(im.shape[0]): 
        for column in range(im.shape[1]):
            avg_contrasts.append(calculate_average_contrast_with_other_pixels(im, im[row][column]))
            printProgressBar(row*im.shape[0]+column+1, im.shape[0]*im.shape[1])
    return np.mean(avg_contrasts), np.var(avg_contrasts)