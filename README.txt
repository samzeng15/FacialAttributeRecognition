Examination of Facial Attribute Recognition methods using Transfer Learning on DenseNet with the CelebA Dataset


Facial attribute recognition is a commonly studied area in image classification, with significant
amounts of prior research and data available. This report details the experimentation and fine tuning of a
multi-label classification neural network optimised for facial attribute recognition over the public domain dataset "CelebA", featuring over 200,000
cropped photos of celebrity faces. Each face has been labelled with their respective identified features representing a multi-label dataset with 40 classes.

The full paper details the experimentation process as well as further justification on decisions made in developing this project.

To run the code, you will need the following 3 files:

1. The folder of images: img_align_celeba
2. list_attr_celeba.txt
3. list_eval_partition.txt

You can download these from the CelebA Dataset website: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html 

These 3 are required for the model data loading phase. 

The best performing model can be downloaded from: https://www.dropbox.com/sh/yzl6gp796f0xuqq/AAC713RV5LARnJPvSiuavPLTa?dl=0


CODE ANALYSIS:

The 2 .py files for the code are data_pipeline.py and celeb_ai.py, representing the code to produce the best performing model as above.

data_pipeline.py conducts the preprocessing steps required for the model to run, and is essentially a set of helper processing functions.

To actually run the training, simply run celeb_ai.py with python in a terminal, ensuring ALL dependencies are installed. Please see
the dependencies section below for more details, or just perform "pip install -r requirements.txt" from the requirements.txt file in the folder.

Adjust the hyperparameters from within the methods of either py file. Note: by default, celeb_ai will run using the fine-tuning method,
running run_fine_tuned_model(). 

It will then train for the specified epochs/fine-tuning steps, before outputing the evaluation plots and then finally returning the
model accuracy. 

The file tf-gpu-test.py can be executed to see if tensorflow has been correctly installed for gpu training. This is very much
recommended, and instructions are not provided here. 

DEPENDENCIES: 

Please ensure that all dependencies are installed. You can do this quickly by using "pip install -r requirements.txt"

They are required for the code to run. Please see the requirements.txt file to see
if you are missing any dependencies. This file was automatically generated using the pipreqs package, so in the unlikely event that these
requirements are not conclusive, please ensure additional dependencies are installed before running the code. 

