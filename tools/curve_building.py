#! /usr/bin/python
import sys
import umap
import umap.plot
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util

print("Importing Embeddings Model...")
model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device='cuda') # TODO investigate changing the model
print(" Done !")

def txt2emb(text: str, windowSize: int, isTensor: bool=True):
    """
    Returns curve embeddings 
    """
# divide given text in a word list 
    wordedText = text.split()
    nbFullWindows = len(wordedText) // windowSize
    # Cuts the given text into a sliding window of `windowSize` words
    sliding_window = [" ".join(wordedText[i:i+windowSize]) for i in range(0, nbFullWindows, windowSize)]
    sliding_window += [" ".join(wordedText[nbFullWindows:])]

    # create the embeddings from the model
    embeddings = model.encode(sliding_window, convert_to_tensor=isTensor)
    return embeddings

def cosine_2by2(embeddings):
    """
    builds a list detailing the 'evolution' of the sementic similarity
    ie the cosine score of the embeddings of two neighbor windows.
    """
    cos_score = []
    for i in range(len(embeddings) - 1):
        cos_score.append(util.cos_sim(embeddings[i], embeddings[i+1]))
    return cos_score        

def distance_2x2(embeddings):
    dist = []
    for i in range(len(embeddings)-1):
        dist += [embeddings[i] - embeddings[i+1]]

def projection_and_plot(embeddings, title="", show = False):
    """
    Project the embeddings in 2d and 3d using umap and plot it
    """
    ## projecting the embeddings with UMAP
    reducer2d = umap.UMAP()
    proj2d = reducer2d.fit_transform(embeddings)
    reducer3d = umap.UMAP(n_components=3)
    proj3d = reducer3d.fit_transform(embeddings)

    ## Plotting of 2d and 3d curves
    # width twice as big as height
    fig = plt.figure(figsize=plt.figaspect(.5))
    fig.suptitle(title, fontsize=24)

    ax2d = fig.add_subplot(1,2, 1)
    ax3d = fig.add_subplot(1,2, 2, projection='3d')
    ax2d.plot(proj2d[:,0], proj2d[:,1])
    beg_curve, = ax2d.plot(proj2d[0,0], proj2d[0,1], 'bx')
    end_curve, = ax2d.plot(proj2d[-1,0], proj2d[-1,1], 'bo')

    ax3d.plot(proj3d[:,0], proj3d[:,1], proj3d[:,2])
    ax3d.plot(proj3d[0,0], proj3d[0,1], proj3d[0,2], 'bx')
    ax3d.plot(proj3d[-1,0], proj3d[-1,1], proj3d[-1,2], 'bo')

    fig.legend([beg_curve, end_curve], ['Beginning', 'End'])
    if title != "":
        fig.savefig("./figures/" + title + ".pdf")
    else:
        fig.savefig("./figures/curve.pdf")

    if show:
        plt.show()

def characterizers(embeddings):
    if len(embeddings) < 3:
        print("Text not long enough.")
    else :
        sumAngle = 0
        sumDist = 0
        for i in range(len(embeddings) - 2):
            P1 = np.array(embeddings[i])
            P2 = np.array(embeddings[i + 1])
            P3 = np.array(embeddings[i + 2])

            sumAngle += np.arccos( np.dot(P2 - P1, P3 - P2) / np.linalg.norm(P1 - P2)/np.linalg.norm(P3 - P2) )
            sumDist += np.linalg.norm(P2- P1)

        meanAngle = sumAngle / (len(embeddings) - 2)
        meanDistance = (sumDist + np.linalg.norm(P3-P2)) / (len(embeddings) - 1)

        return meanAngle, meanDistance

if __name__ == "__main__":
    """ 
    Takes a text as first input to create its projected stylistic curve in 2d and 3d
    """
    # TODO Q: do we need to clean the sentences ie strip them of most common words ?
    with open(sys.argv[1]) as f:
       text = f.read()

    print("computing embeddings ...")
    embeddings = txt2emb(text, 5, False)

    ## Plotting of 2d and 3d curves
    title =""
    if len(sys.argv) == 3:
        title = sys.argv[2]
    
    MeanCurv, MeanDistance = characterizers(embeddings)
    print("Mean Distance : ", MeanDistance, "MeanCurvature (rad) : ", MeanCurv)
    projection_and_plot(embeddings, title, True)
