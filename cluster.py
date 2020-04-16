import numpy as np
import matplotlib.pyplot as plt
import json

data = json.load(open('contrast_data.json'))
means_hdr = [i[0] for i in data['HDR']]
means_sdr = [i[0] for i in data['SDR']]
plt.scatter(np.arange(len(means_hdr)), means_hdr)
plt.scatter(np.arange(len(means_sdr)), means_sdr)
plt.show()