import os
import joblib
import pandas as pd
from curve_building import *
from sklearn.neural_network import MLPClassifier


## create the style curve of all the texts in the dataset
windowSize = 3
nbEmbs = 20

if not os.path.exists("train_data.pkl"):
# open a csv file and load the two column as variables
    data = pd.read_csv("fake_papers_train_part_public.csv")
    texts_vectors = dataset_vectorized_curve(data["text"], windowSize, nbEmbs)
    joblib.dump(texts_vectors, "train_data.pkl", compress=3)
else:
    texts_vectors = joblib.load("train_data.pkl")


## Train the classifier
print("Training the Classifier ...")
clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(384*nbEmbs), random_state=1) # 384 is the emb size for sbert
clf.fit(texts_vectors, data['fake'])
print("Done!")

# Save classifier
joblib.dump(clf, "clf_20e_3w.pkl", compress=3)

# open a test text and output the result
with open('testdocs/attention_intro.txt', 'r') as f:
    testCurve = txt2emb(f.read(), windowSize, False)
testCurveVec = np.zeros((1,384*nbEmbs))
# vectorize the curve and feed it to the clf
for j in range(min(len(testCurve), nbEmbs)):
    testCurveVec[384*j:384*(j+1)] += testCurve[j]
print( clf.predict(testCurveVec) )
