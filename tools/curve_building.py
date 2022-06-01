#! /usr/bin/python
import sys
import umap
import umap.plot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device='cuda') # TODO investigate changing the model

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

if __name__ == "__main__":
    """ 
    Takes a text as first input to create its projected stylistic curve in 2d and 3d
    """
    # TODO Q: do we need to clean the sentences ie strip them of most common words ?
    with open(sys.argv[1]) as f:
       text = f.read()

    print("computing embeddings ...")
    embeddings = txt2emb(text, 5, False)

    ## projecting the embeddings with UMAP
    reducer2d = umap.UMAP()
    proj2d = reducer2d.fit_transform(embeddings)
    reducer3d = umap.UMAP(n_components=3)
    proj3d = reducer3d.fit_transform(embeddings)

    ## Plotting of 2d and 3d curves
    fig = plt.figure(figsize=plt.figaspect(.5))
    if len(sys.argv) == 3:
        fig.suptitle(sys.argv[2], fontsize=24)

    
    ax2d = fig.add_subplot(1,2, 1)
    ax3d = fig.add_subplot(1,2, 2, projection='3d')
    ax2d.plot(proj2d[:,0], proj2d[:,1])
    beg_curve, = ax2d.plot(proj2d[0,0], proj2d[0,1], 'bx')
    end_curve, = ax2d.plot(proj2d[-1,0], proj2d[-1,1], 'bo')

    ax3d.plot(proj3d[:,0], proj3d[:,1], proj3d[:,2])
    ax3d.plot(proj3d[0,0], proj3d[0,1], proj3d[0,2], 'bx')
    ax3d.plot(proj3d[-1,0], proj3d[-1,1], proj3d[-1,2], 'bo')

    fig.legend([beg_curve, end_curve], ['Beginning', 'End'])
    plt.show()
