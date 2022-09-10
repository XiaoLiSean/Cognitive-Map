#SeanNet: Semantic Understanding Network for localization Under Object Dynamics

We aim for domestic robots to perform long-term indoor service. Under the object-level scene dynamics induced by daily human activities, a robot needs to robustly localize itself in the environment subject to scene uncertainties. Previous works have addressed visual-based localization in static environments, yet the object-level scene dynamics challenge existing methods for the long-term deployment of the robot. This paper proposes a SEmantic understANding Network (SeanNet) architecture that enables an effective learning process with coupled visual and semantic inputs. With a dataset that contains object dynamics, we propose a cascaded contrastive learning scheme to train the SeanNet for learning a vector scene embedding. Subsequently, we can measure the similarity between the current observed scene and the target scene, whereby enables robust localization under object-level scene dynamics. In our experiments, we benchmark SeanNet against state-of-the-art image-encoding networks (baselines) on scene similarity measures. The SeanNet architecture with the proposed training method can achieve an 85.02\% accuracy which is higher than baselines. We further integrate the SeanNet and the other networks as the localizers into a visual navigation application. We demonstrate that SeanNet achieves higher success rates compared to the baselines.

## Build environment
Generally, our codebase has been tested functional on platforms of macOS, Ubuntu 18, and Ubuntu 20. To set up the conda virtual environment and build up the required packages, please download the code and run the command ```conda env create -f environment.yml``` in the root folder. After successfully building up the environment, there are a few files to construct for global usage. 

## Prepare System Required Files
1. ```mkdir AI2THOR_info```
2. ```mkdir 3rdparty```
3. Download "glove.42B.300d.txt" from the GloVe official website (https://nlp.stanford.edu/projects/glove/) to the folder ```3rdparty```
4. Excute program ```python prepare_files.py``` to generate all needed filed
5. Test if all the files are successfully generated using ```python test_build.py```
If the AI2THOR window pops out with scene graph alongside the image, then the environment is successfully built and the required files are generated

## Dataset and Training
It usually takes 5-8 hours to collect the dataset for training the navigation and localization networks
1. To collect a raw dataset of images and scene graphs ```python collect_data.py --collect_partition```
2. To generate triplet training and validation dataset for **Localization** ```python collect_data.py --regenerate_RNet```
3. To generate pair-wise threshold testing dataset from previous triplet validation dataset for **similarity measure** ```python collect_data.py --gen_pair_in_val```
4. To generate training and validation dataset for **Navigation** ```python collect_data.py --regenerate_NaviNet```

### Train and test localization networks
1. Train SeanNet-based localization network: ```python discrete_RNet_pipeline.py --train --rnet```
2. Generate heatmaps (similarity score versus positional deviation in x and y direction) as in the paper to obtain the localization threshold ```python discrete_RNet_pipeline.py --heatmap --rnet```
3. Train Benchmarks-based localization network: ```python discrete_RNet_pipeline.py --train --benchmark --name benchmarkName``` where you can replace the variable benchmarkName with benchmarks with name ['resnet50', 'vgg16', 'googlenet', 'resnext50_32x4d']
4. Generate heatmaps for benchmarks ```python discrete_RNet_pipeline.py --heatmap --benchmark --name benchmarkName```
5. Use your localization thresholds epsilons that picked in steps 2,4. Fill those values in function ```is_localized()``` in file ```/Network/retrieval_network/retrieval_network.py```
6. Test SeanNet-based localization network: ```python discrete_RNet_pipeline.py --test --rnet```
7. Test Benchmarks-based localization network: ```python discrete_RNet_pipeline.py --test --benchmark --name benchmarkName```


### Train and test navigation networks
1. Train SeanNet-based navigation network: ```python discrete_Navi_pipeline.py --train --rnet```
2. Train Benchmarks-based navigation network: ```python discrete_Navi_pipeline.py --train --benchmark --name benchmarkName```
3. Test SeanNet-based navigation network: ```python discrete_Navi_pipeline.py --test --rnet```
4. Test Benchmarks-based navigation network: ```python discrete_Navi_pipeline.py --test --benchmark --name benchmarkName```

## Similarity Measure with SeanNet in Simulation

![tripletEx1](https://user-images.githubusercontent.com/58400416/189503366-e4f0fe99-3163-4b91-91c8-b7e6a26dcc85.png)

![tripletEx2](https://user-images.githubusercontent.com/58400416/189503376-91f98ae9-6514-42c5-9312-533fe2574de7.png)

![tripletEx3](https://user-images.githubusercontent.com/58400416/189503380-d51ec0a6-3735-4c71-ae17-6dca03c7d297.png)

## Similarity Measure with SeanNet in Real World

![realWorldEx1](https://user-images.githubusercontent.com/58400416/189503432-1f2a8cd9-3663-4eca-a26d-534c7ac9561f.png)

![realWorldEx2](https://user-images.githubusercontent.com/58400416/189503435-7e4f2997-80d8-41b9-9431-d139a222ede6.png)


## Navigation demo

After successfully training the localization and navigation networks with corresponding network architectures, you can run ```python visual_navi_demo.py``` to obtain a visual navigation demonstration with trainned networks for visual localization.

https://user-images.githubusercontent.com/58400416/189503280-cd64943f-2771-4940-9612-02ea132fc8ab.mp4


Please cite as: 
```
@article{li2021seannet,
  title={SeanNet: Semantic Understanding Network for Localization Under Object Dynamics},
  author={Li, Xiao and Du, Yidong and Zeng, Zhen and Jenkins, Odest Chadwicke},
  journal={arXiv preprint arXiv:2110.02276},
  year={2021}
}
```
