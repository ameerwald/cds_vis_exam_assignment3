
# Assignment 3 - Using pretrained CNNs for image classification

This assignment can be found at my github [repo](https://github.com/ameerwald/cds_vis_exam_assignment3).


For this assignment, the data is a fashion dataset, *Indo fashion* taken from this [Kaggle dataset](https://www.kaggle.com/datasets/validmodel/indo-fashion-dataset). 

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
| Utils  | Unfinished preprocessing script.



## To run the scripts 
As the dataset is too large to store in my repo, use the link above to access the data. Download and unzip the data. Then when running this, if on Ucloud for example, create a folder called  ```data``` along with the other folders in the assignment 3 repo. Then the code will run without making any changes. If the data is placed elsewhere, the path should be updated in the code. 

From the command line, run the following chunk of code. 
``` 
bash setup.sh
bash run.sh
```

This has been run on an ubuntu system on ucloud and therefore could have issues when run another way.
