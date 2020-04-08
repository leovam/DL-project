### implementation of 1D_CNN ###
use the default hyperparameter in the code
use Sheng's preprocessed TCGA data

#### models ####
1. OneDCNN34.py: 33 cancer types + normal samples
2. OneDCNN33.py: only 33 cancer types used
3. compare33_34.py: run the two models above 10 times and compare the results

#### test ####
1. run test on each model: python OneDCNN*.py
2. compare two models: python compare33_34.py
(make sure you run the model on GPU)

#### result ####
![](box_compare.png)