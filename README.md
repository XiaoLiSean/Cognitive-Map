SeanNet: Semantic Understanding Network for localization Under Object Dynamics

## Build environment
Generally, our codebase is tested functional on platforms of macOS, Ubuntu 18, and Ubuntu 20. To set up the conda virtual environment and build up the required packages, please download the code and run the command ```conda env create -f environment.yml``` in the root folder. After successfully building up the environment, there are a few files to construct for global usage. 

## Prepare System Required Files
1. ```mkdir AI2THOR_info```
2. ```mkdir 3rdparty```
3. Download "glove.42B.300d.txt" from the official website (GloVe word vector embedding @ https://nlp.stanford.edu/projects/glove/)  
4. Excute file to generate all needed filed ```python prepare_files.py```
5. Test if all the files are successfully generated ```python test_build.py```
If the AI2THOR window pops out with scene graphs alongside the image, then the environment is successfully built and the required files are generated

## Dataset and Training
1. Run XXX to collect data
2. Run XXX to train networks

## Navigation demo
1. Run navigation demo using XXX
