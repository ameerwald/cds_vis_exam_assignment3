
# Assignment 3 - Using pretrained CNNs for image classification

## Github repo link 

This assignment can be found at my github [repo](https://github.com/ameerwald/cds_vis_exam_assignment3).

## The data

For this assignment, the data is a fashion dataset, *Indo fashion* taken from this [Kaggle dataset](https://www.kaggle.com/datasets/validmodel/indo-fashion-dataset). 

## Assignment description 

The instructions for the assignment are as follows:

- Write code which trains a classifier on this dataset using a *pretrained CNN like VGG16*
- Save the training and validation history plots
- Save the classification report


## Repository 

| Folder         | Description          
| ------------- |:-------------:
| Data      | I have hidden this folder because the dataset in too large to push to Github 
| Notes  | Notes to help me figure out how to run the code in a py script 
| Out  | Classification Report and history plot  
| Src  | Py script  
| Utils  | Preprocessing script with utility functions



## To run the scripts 
As the dataset is too large to store in my repo, use the link above to access the data. Download and unzip the data. Then create a folder called  ```data``` within the assignment 3 folder, along with the other folders in the repo. Then the code will run without making any changes. If the data is placed elsewhere, then the path should be updated in the code.

1. Clone the repository, either on ucloud or something like worker2
2. From the command line, at the /cds_vis_exam_assignment3/ folder level, run the following chunk of code. This will create a virtual environment, install the correct requirements, run the following lines of code. 

This will create a virtual environment, install the correct requirements.
``` 
bash setup.sh
```
While this will run the scripts and deactivate the virtual environment when it is done. 
```
bash run.sh
```
This has been tested on an ubuntu system on ucloud and therefore could have issues when run another way.

## Discussion of Results 
Overall the accuracy is quite high, 65% as seen in the classification report in the ```out``` folder. Not all categories perform well though, with the lowest being dupattas which the model only correctly identified 29% of the time. However this model was only trained over 2 epoches due to the computational power and time further training requires. If trained over more epochs, the overall accurary could improve including within the specific categories that are not performing well with this model. 
