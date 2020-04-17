import matplotlib.pyplot as plt
import json


neg_count = 0
pos_count = 0
data = json.load(open("luminance_data.json"))
for item in data:
    for entry in data[item]:
        name, luminance_data = entry
        diff = luminance_data[0]
        if diff < 0:
            pos_count += 1
        else:
            neg_count += 1

print(pos_count, neg_count)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
langs = ['HDR less than SDR', 'SDR less than HDR']
students = [pos_count, neg_count]
ax.bar(langs,students)
plt.show()