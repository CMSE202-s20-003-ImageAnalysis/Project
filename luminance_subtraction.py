from contrast_measurer import image_contrast_between
import os
import json

ans = {}
directories = [item for item in os.listdir() if '.' not in item and '__' not in item]

counter = 204
for directory in directories:
    files = [item for item in os.listdir(f'./{directory}') if 'Fractal' not in item and ('HDR' in item or 'SDR' in item)]
    for i in files:
        if 'HDR' in i:
            start = int(i.index('-')+1)
            end = int(i.index('.'))
            sdr_name = 'SDR-{}.jpg'.format(i[start:end])
            if sdr_name in files:
                try:
                    luminance_diff = image_contrast_between(f'./{directory}/{i}', f'./{directory}/{sdr_name}')
                    ans.setdefault(directory, []).append((i, luminance_diff))
                    counter -= 1
                    print(f'{counter} files left...')
                except:
                    print(f'Error in /{directory}/{i}')
                

json.dump(ans, open("luminance_data.json", 'w'), indent=4)
