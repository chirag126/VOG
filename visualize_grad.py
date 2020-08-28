import os
import ipdb
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt


def plot_grid(vog_list, out_name):
	img_str = ''
	for ind, i in enumerate(vog_list):
		img_str += './class_701/weight_00000/img_{:05d}.jpg '.format(i[0])
	os.system('montage -quiet {} -tile 3x3 -geometry +0+0 {}'.format(img_str, out_name))
	
# Image indexes
img_ind = [int(f.split('.')[0].split('_')[-1]) for f in os.listdir('./class_701/weight_00000/') if f.startswith('img')]
# ipdb.set_trace()

vog_score = {}
for ind in img_ind:
	weight_files = sorted(os.listdir('./class_701/'))[4:]
	# plt.figure()
	# plt.subplot(171); plt.imshow(imread('./class_701/weight_00000/img_{:05d}.jpg'.format(ind))); plt.axis('off')
	
	mean_grad = np.zeros((224, 224))
	for i, weight in enumerate(weight_files):
	        grad = np.load('./class_701/{}/grad_{:05d}.npy'.format(weight, ind))
	        mean_grad += grad
	mean_grad /= len(weight_files)

	temp_grad = 0
	for i, weight in enumerate(weight_files):
		grad = np.load('./class_701/{}/grad_{:05d}.npy'.format(weight, ind))
	#	plt.subplot(1,7,i+2);plt.imshow(grad); plt.axis('off')
		temp_grad += np.mean((grad - mean_grad)**2)

	vog = np.sqrt(temp_grad/len(weight_files))

	# plt.savefig('demo.pdf', bbox_inches='tight', pad=0)
	vog_score[ind] = vog

# ipdb.set_trace()
# print(sorted(vog_score.items(), key=lambda x: x[1], reverse=True))

## Top-9 Early stage
#lowest_9_early = sorted(vog_score.items(), key=lambda x: x[1], reverse=False)[:9]
#highest_9_early = sorted(vog_score.items(), key=lambda x: x[1], reverse=True)[:9]
#plot_grid(lowest_9_early, 'low_9_early.jpg')
#plot_grid(highest_9_early, 'high_9_early.jpg')

# Top-9 Late stage
lowest_9_late = sorted(vog_score.items(), key=lambda x: x[1], reverse=False)[:9]
highest_9_late = sorted(vog_score.items(), key=lambda x: x[1], reverse=True)[:9]
plot_grid(lowest_9_late, 'low_9_late.jpg')
plot_grid(highest_9_late, 'high_9_late.jpg')

