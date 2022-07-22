"""
This program reads a csv file containing text and fake column, run all of the
texts through the classifier and outputs a csv file with the original 'fake' column
and the score that the classifier gave to each text.
|  text  |  fake  | ---> |  text  |  fake  |  score  |  
"""
import pandas as pd
from sys import argv
from tools.curve_building import *
from sklearn.neural_network import MLPClassifier
import joblib
import re

if bool(re.search("_\d+e_\d+w", argv[1])):
    foo = re.findall("_\d+", argv[1])
    nbEmbs = int(foo[0].strip('_'))
    windowSize = int(foo[1].strip('_'))
else:
    sys.exit("Input classifier badly formatted. Please rename of check filename.")

clf = joblib.load(argv[1])
# read a csv file with columns text & artificial
data = pd.read_csv(argv[2])
# open a test text and output the result
curve_list = dataset_vectorized_curve(data['text'], windowSize, nbEmbs)
results = clf.predict_proba(curve_list)[:,0]

# Creating and saving data as csv file
data['score'] = results
data.to_csv(sys.argv[3])
