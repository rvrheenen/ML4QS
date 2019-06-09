##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 3                                               #
#                                                            #
##############################################################

from util.VisualizeDataset import VisualizeDataset
from Chapter3.OutlierDetection import DistributionBasedOutlierDetection
from Chapter3.OutlierDetection import DistanceBasedOutlierDetection
import copy
import pandas as pd
import numpy as np

def plot_chauvenet(dataset):
    for col in outlier_columns:
        for c in [1, 10, 100]:
            dataset = OutlierDistr.chauvenet(dataset, col, c)
            DataViz.plot_binary_outliers(dataset, col, col + '_outlier')

def plot_mixture(dataset):
    for col in outlier_columns:
        for k in range(3,30,3):
            dataset = OutlierDistr.mixture_model(dataset, col, k)
            DataViz.plot_dataset(dataset, [col, col + '_mixture'], ['exact','exact'], ['line', 'points'])



# Let is create our visualization class again.
DataViz = VisualizeDataset()

# Read the result from the previous chapter, and make sture the index is of the type datetime.
dataset_path = './intermediate_datafiles/'
try:
    dataset = pd.read_csv(dataset_path + 'chapter2_result.csv', index_col=0)
except IOError as e:
    print('File not found, try to run previous crowdsignals scripts first!')
    raise e

dataset.index = dataset.index.to_datetime()

# Compute the number of milliseconds covered by an instance based on the first two rows
milliseconds_per_instance = (dataset.index[1] - dataset.index[0]).microseconds/1000

# Step 1: Let us see whether we have some outliers we would prefer to remove.

# Determine the columns we want to experiment on.
outlier_columns = ['acc_phone_x', 'light_phone_lux']

# Create the outlier classes.
OutlierDistr = DistributionBasedOutlierDetection()
OutlierDist = DistanceBasedOutlierDetection()

plot_chauvenet(dataset)
plot_mixture(dataset)