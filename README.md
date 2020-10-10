## Estimating Example Difficulty using Variance of Gradients

This repository contains source code necessary to reproduce some of the main results in [the paper](https://arxiv.org/abs/2008.11600):

**If you use this software, please consider citing:**
    
    @article{agarwal2020estimating, 
    title={Estimating Example Difficulty using Variance of Gradients},
    author={Agarwal, Chirag and Hooker, Sara},
    journal={arXiv preprint arXiv:2008.11600},
    year={2020}
    }
    
## 1. Setup

### Installing software
This repository is built using a combination of TensorFlow and PyTorch. You can install the necessary libraries by pip installing the requirements text file `pip install -r ./requirements_pytorch.txt` & `pip install -r ./requirements_tf.txt`

## 2. Usage
### Toy experiment
[toy_script.py](toy_script.py) is the script for running toy dataset experiment. You can analyze the training/testing data at diffferent stages of the training, viz. Early, Middle, and Late, using the flags `split` and `mode`. The `vog_cal` flag enables visualizing different versions of VOG scores such as the raw score, class normalized, or the absolute class normalized scores. 

#### Examples
Running `python3 toy_script.py --split test --mode early --vog_cal normalize` generates the toy dataset decision boundary figure along with the relation between the perpendicular distance of individual points from the decision boundary and the VOG scores. The respective figures are:

<p align="center">
    <img src="figures/toy_dataset_decision_boundary.jpg" width=250px>
    <img src="figures/test_early_normalize.jpg" width=250px>
</p>
<p align="left"><i>Left: The visualization of the toy dataset decision boundary with the testing data points. The Multiple Layer Perceptron model achieves 100% training accuracy. Right: The scatter plot between the Variance of Gradients (VoGs) for each testing data point and their perpendicular distance shows that higher scores pertain to the most
challenging examples (closest to the decision boundary)</i></p>

### ImageNet
The main scripts for the ImageNet experiments are in the `./imagenet/` folder. Before calculating the VOG scores you would need to store the gradients of the respective images in the `./scripts/train.txt/` file using model snapshots. For demonstration purpose, we have shared the model weights at snapshot `32000`. To store the gradients for the imagenet dataset (stored as <path>/imagenet_dir/train), we run the shell script [train_get_gradients.sh](train_get_gradients.sh) like:
`source train_get_gradients.sh 32000 ./imagenet/train_results/ 9 ./scripts/train.txt/`



[LIME_test.sh](LIME_test.sh): 
Generating the attribution map for the class "kuvasz" using LIME and LIME-G algorithm.
* Running `source LIME_test.sh` produces this result:

<p align="center">
    <img src="output/test_LIME.jpg" width=750px>
</p>
<p align="center"><i>(left-->right) The real image followed by five random intermediate perturbed images and the resultant attribution map for LIME (top) and LIME-G (bottom). For each intermediate perturbed image, the top and bottom row labels shows the target and top-1 class predictions with their respective probabilities.</i></p>

[MP_test.sh](MP_test.sh): 
Generating the attribution map for the class "freight car" using MP and MP-G algorithm.
* Running `source MP_test.sh` produces this result:

<p align="center">
    <img src="output/test_MP.jpg" width=750px> 
</p>
<p align="center"><i>(left-->right) The real image followed by five random intermediate perturbed images and the resultant attribution map for MP (top) and MP-G (bottom). For each intermediate perturbed image, the top and bottom row labels shows the target and top-1 class predictions with their respective probabilities.</i></p>

## 4. Licenses
Note that the code in this repository is licensed under MIT License, but, the pre-trained condition models used by the code have their own licenses. Please carefully check them before use. 

## 5. Questions?
If you have questions/suggestions, please feel free to [email](mailto:chiragagarwall12@gmail.com) or create github issues.     
