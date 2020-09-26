import os
import ipdb
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 11, 'font.weight': 'bold'})
plt.rc('font', family='sans-serif')


load_path = 'train_results'
imagenet_classes = np.genfromtxt('./imagenet_classes.txt', dtype=str, delimiter=':')

def compare_c_score_and_vog(vog, c_score_file):
    ipdb.set_trace()
    filenames = c_score_file['filenames']
    filenames = [ff.decode('utf-8') for ff in filenames]
    c_score = c_score_file['scores']
    image_names = np.genfromtxt('./scripts/train.txt', delimiter=' ', dtype=str)[:10000, 0]
    imagenames = [ff.split('/')[-1] for ff in image_names] 
    selected_c_scores = [c_score[filenames.index(ff)] for ff in imagenames]
    plt.figure()
    plt.scatter(vog, selected_c_scores, linewidths=1.5, edgecolors='k', c='b', alpha=0.6
            )
    plt.xlabel('VOG score')
    plt.ylabel('C-score')


def plot_grid(vog_list, ground_truth, out_name, pr):
    for lab in [701]:  # range(1000):
        img_str = ''
        count = 0
        for ind in vog_list:
            if ground_truth[ind] == lab and count < 25:
                pr_label = imagenet_classes[pr[ind]][1][2:-1].split(',')[0] 
                img_str += '-label "{}" ./{}/weight_32000/img_{:05d}.jpg '.format(pr_label, load_path, ind)
                count += 1
        os.system('montage -quiet {} -tile 5x5 -geometry +0+0 -pointsize 33 ./imagenet_samples/{:03d}_{}'.format(img_str, lab, out_name))

# Image indexes
img_ind = sorted([int(f.split('.')[0].split('_')[-1]) for f in os.listdir('./{}/weight_32000/'.format(load_path)) if f.startswith('grad')])
ground_truth = np.load('./{}/weight_32000/ground_truth.npy'.format(load_path))
c_score = np.load('./imagenet-cscores-with-filename.npz', allow_pickle=True)

# List of weight files
snapshot_list=['early', 'middle', 'late', 'complete']
for snapshot in snapshot_list:
    if snapshot == 'early':
        weight_files = sorted(os.listdir('./{}/'.format(load_path)))[:3]
    elif snapshot == 'middle':
        weight_files = sorted(os.listdir('./{}/'.format(load_path)))[3:6]
    elif snapshot == 'late':
        weight_files = sorted(os.listdir('./{}/'.format(load_path)))[6:]
    elif snapshot == 'complete':
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
    for ind in img_ind:
        temp_grad = []
        for i, weight in enumerate(weight_files):
            temp_grad.append(np.load('./{}/{}/grad_{:05d}.npy'.format(load_path, weight, ind)))
        mean_grad = np.sum(np.array(temp_grad), axis=0)/len(temp_grad)
        vog=np.mean(np.sqrt(sum([(mm-mean_grad)**2 for mm in temp_grad])/len(temp_grad)))
        vog_stats.append(vog)
        vog_class_stats[ground_truth[ind]].append(vog)
    print('Done!!')

    ## Normalized VOG score
    #print('=== Normalizing the VOG scores using class-wise statistics ===')
    #normalized_vog=[]
    #for ind in img_ind:
    #    mu = np.mean(vog_class_stats[ground_truth[ind]])
    #    std = np.std(vog_class_stats[ground_truth[ind]])
    #    normalized_vog.append(((vog_stats[ind]-mu)/std))
    #print('Done!!')

    # compare_c_score_and_vog(vog_stats, c_score)
    # plt.savefig('compare_c_score_vog_scatter_{}.pdf'.format(snapshot), bbox_inches='tight', pad=0)
    # continue

    normalized_vog = vog_stats
    np.save('imagenet_train_vog_{}.npy'.format(snapshot), normalized_vog)

    # Calculate class-wise accuracy
    class_wise_acc = np.array([
        np.mean(np.mean(classifier_stats[ground_truth==i, 1])) for i in range(1000)])

    ## Sorting the VOG scores
    lowest_vog = sorted(range(len(normalized_vog)), key=lambda k: normalized_vog[k], reverse=False)
    highest_vog = sorted(range(len(normalized_vog)), key=lambda k: normalized_vog[k], reverse=True)

    # For visualizing the images
    plot_grid(lowest_vog, ground_truth, '{}_lowest_vog.jpg'.format(snapshot), pred_label)
    plot_grid(highest_vog, ground_truth, '{}_highest_vog.jpg'.format(snapshot), pred_label)

    def calculate_top_10_percentile_error(x, pr):
        # ipdb.set_trace()
        mag_x = [m for m in x]
        bottom_10_percentile = np.percentile(mag_x, 10)
        bottom_10_ind = [ind for ind in range(len(mag_x)) if mag_x[ind] < bottom_10_percentile]
        bottom_10_misclassify = [mm for mm in bottom_10_ind if (1 != pr[mm])]
        top_10_percentile = np.percentile(mag_x, 90)
        top_10_ind = [ind for ind in range(len(mag_x)) if mag_x[ind] > top_10_percentile]
        top_10_misclassify = [mm for mm in top_10_ind if (1 != pr[mm])]
        bottom_10_ind_acc = 100*(len(bottom_10_misclassify)/len(bottom_10_ind))
        top_10_ind_acc = 100*(len(top_10_misclassify)/len(top_10_ind))
        all_test_acc = 100*(len([mm for mm in range(50000) if (1 != pr[mm])])/50000)

        print(bottom_10_ind_acc, all_test_acc, top_10_ind_acc)
        plt.figure()
        plt.bar(np.arange(3), [bottom_10_ind_acc, all_test_acc, top_10_ind_acc])
        plt.xticks(np.arange(3), ('bottom-10', 'all-test', 'top-10'))
        plt.ylabel('% top-1 test set error')
        plt.yticks(np.arange(0, 51, 10))
        plt.savefig('bar_{}.pdf'.format(snapshot), bbox_inches='tight', pad=0)
        
        # Trend of VOG scores with misclassified samples
        misclassify = []
        for i in range(10):
            percentile_1, percentile_2 = np.percentile(mag_x, i*10), np.percentile(mag_x, (i+1)*10)
            ind = [ind for ind in range(len(mag_x)) if (mag_x[ind] < percentile_2) and (mag_x[ind] > percentile_1)]
            misclassify.append(100*len([mm for mm in ind if (1 != pr[mm])])/len(ind))
        plt.figure()
        plt.plot(np.arange(5, 99, 10), misclassify, '-o', markersize=10, markeredgewidth=1.5, markeredgecolor='k')
        plt.ylabel('% top-1 test set error')
        plt.xlabel('VOG percentile range')
        plt.xticks(np.arange(0, 101, 10))
        plt.yticks(np.arange(0, 101, 10))
        plt.savefig('imagenet_unnormalized_error_plot_{}.pdf'.format(snapshot), bbox_inches='tight', pad=0)

    print('Completed UNNORMALIZED analysis')
    calculate_top_10_percentile_error(normalized_vog, classifier_stats[:, 1])
