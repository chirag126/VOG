import os
import ipdb
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 11, 'font.weight': 'bold'})
plt.rc('font', family='sans-serif')


# Path pointing to the gradient results of ImageNet
load_path = 'train_results'

# ImageNet classes file for labelling
imagenet_classes = np.genfromtxt('./imagenet_classes.txt', dtype=str, delimiter=':')

def calculate_top_10_percentile_error(x, pr):        
    # Trend of VOG scores with misclassified samples
    misclassify = []
    for i in range(2):
        percentile_1, percentile_2 = np.percentile(x, i*50), np.percentile(x, (i+1)*50)
        ind = [ind for ind in range(len(x)) if (x[ind] < percentile_2) and (x[ind] > percentile_1)]
        misclassify.append(100*len([mm for mm in ind if (1 != pr[mm])])/len(ind))
    plt.figure()
    plt.plot(np.arange(25, 100, 50), misclassify, '-o', markersize=10, markeredgewidth=1.5, markeredgecolor='k')
    plt.ylabel('% top-1 test set error')
    plt.xlabel('VOG percentile range')
    plt.xticks(np.arange(0, 101, 10))
    plt.savefig('imagenet_error_plot_{}.jpg'.format(snapshot), bbox_inches='tight')


def plot_grid(score_x, vog, dataset_classes, x_gt, x_pr, label_type):
    count = 0
    for ind in score_x:
        img_str = '-label "{:.4f}\nGT: {}\nPT: {}" ./{}/weight_32000/img_{:05d}.jpg '.format(vog[ind], dataset_classes[x_gt[ind]][1][2:-1].split(',')[0], dataset_classes[x_pr[ind]][1][2:-1].split(',')[0], load_path, ind)
        os.system('montage -quiet {} -tile 1x -geometry +0+0 -pointsize 33 ./{:03d}_{}.jpg'.format(img_str, count, label_type))
        count += 1


# Image indexes
file_ind = sorted([int(f.split('.')[0].split('_')[-1]) for f in os.listdir('./{}/weight_32000/'.format(load_path)) if f.startswith('grad')])
ground_truth = np.load('./{}/weight_32000/ground_truth.npy'.format(load_path))

# List of weight files
snapshot_list = ['late']
for snapshot in snapshot_list:

    weight_files = sorted(os.listdir('./{}/'.format(load_path)))
    print('=== Analyzing Stage: {} ==='.format(snapshot))
    print(weight_files)

    classifier_stats = np.load('./{}/{}/classified_flag.npy'.format(load_path, weight_files[-1]))
    pred_label = np.load('./{}/{}/pred_label.npy'.format(load_path, weight_files[-1]))
    print('=== Calculting VOG scores for eVal set ===')

    # Get class-wise mean gradient for each weight
    mean_grad={}
    count_class={}
    vog_stats=[]
    vog_labels=[]
    vog_class_stats=list(list() for i in range(1000))
    for ind in file_ind:
        temp_grad = []
        for i, weight in enumerate(weight_files):
            temp_grad.append(np.load('./{}/{}/grad_{:05d}.npy'.format(load_path, weight, ind)))
        mean_grad = np.sum(np.array(temp_grad), axis=0)/len(temp_grad)
        vog=np.mean(np.sqrt(sum([(mm-mean_grad)**2 for mm in temp_grad])/len(temp_grad)))
        vog_stats.append(vog)
        vog_class_stats[ground_truth[ind]].append(vog)
    print('VOG calculation done!!')

    ## For this demo script we only have 100 images from different labels. Hence, we comment out the noramlization code
    ## Normalized VOG score
    #print('=== Normalizing the VOG scores using class-wise statistics ===')
    #normalized_vog=[]
    #for ind in file_ind:
    #    mu = np.mean(vog_class_stats[ground_truth[ind]])
    #    std = np.std(vog_class_stats[ground_truth[ind]])
    #    normalized_vog.append(((vog_stats[ind]-mu)/std))
    #print('VOG Normalization done!!')

    normalized_vog = vog_stats

    # Saving VOG numpy file
    np.save('imagenet_train_vog_{}.npy'.format(snapshot), normalized_vog)

    ## Sorting the VOG scores
    lowest_vog = sorted(range(len(normalized_vog)), key=lambda k: normalized_vog[k], reverse=False)
    highest_vog = sorted(range(len(normalized_vog)), key=lambda k: normalized_vog[k], reverse=True)

    # For visualizing the images
    plot_grid(lowest_vog, normalized_vog, imagenet_classes, ground_truth, pred_label, snapshot)

    # Error rate Analysis
    calculate_top_10_percentile_error(normalized_vog, classifier_stats[:, 1])
