import os
import pandas as pd
# tf tools
#scikit-learn
from sklearn.metrics import classification_report
# importing functions from preprocessing script 
import sys
sys.path.append(".")
# import my functions in the utils folder 
from utils.preprocessing import plot_history
from utils.preprocessing import load_data
from utils.preprocessing import load_model
from utils.preprocessing import data_generator
from utils.preprocessing import train_model


def main():
    # load and get labels for the data
    test_data, val_data, train_data, labelNames = load_data()
    # load the model
    model = load_model()
    # generate extra data 
    datagen = data_generator()
    # train the model
    train_images, val_images, test_images, H, predictions = train_model(datagen, train_data, val_data, test_data, model)
    # Save the plotted history over the epochs 
    plot_history(H, 2)
    #dump(plot, os.path.join("out", "plot_history.png"))
    # classification report 
    report = (classification_report(test_images.classes,
                                predictions.argmax(axis=1),
                                target_names=labelNames))
    # save the report 
    with open(os.path.join("out", "classification_report.txt"), "w") as f:
        f.write(report)



if __name__=="__main__":
    main()



