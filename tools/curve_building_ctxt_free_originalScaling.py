#! /usr/bin/python
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("Importing Embeddings Model...")
with open("./glove.6B.300d.txt", 'r') as f:
    l = list(f)

# Create the dic to hold every embeddings. 
gloveDic = {}
embDim = 300
for emb in l:
    # splits the line in the text file : it is of this form 
    # word float float float float ... float \n
    listed_emb = emb.split(' ')
    gloveDic[listed_emb[0]] = np.array([float(s) for s in listed_emb[1:]])
print(" Done !")

print("Creating mean vector...")
meanVec = np.zeros((embDim))
for w in gloveDic.values():
    meanVec += w
meanVec/len(gloveDic)
print(" Done !")

cosSim = lambda x,y : np.dot(x, y) / (np.linalg.norm(x)*np.linalg.norm(y))

def txt2emb(text: str, windowSize: int, isJumping:bool=False, scale:bool = True, meanVec:bool = True):
    """
    Returns curve embeddings for a text or a list of text.
    """
    import wordfreq
    from wordfreq import zipf_frequency
# divide given text in a word list 
    wordedText = wordfreq.tokenize(text, 'en')
    if isJumping:
        nbFullWindows = len(wordedText) // windowSize
        # Cuts the given text into a sliding window of `windowSize` words
        sliding_windows = [" ".join(wordedText[i:i+windowSize]) for i in range(0, len(wordedText), windowSize)]
        sliding_windows += [" ".join(wordedText[nbFullWindows*windowSize + 1:])]
    else:
        nbFullWindows = len(wordedText) - windowSize + 1
        # Cuts the given text into a sliding window of `windowSize` words
        sliding_windows = [" ".join(wordedText[i:i+windowSize]) for i in range(0, nbFullWindows)]

    nonVocGlove = 0 # counter for the amount of words not in Glove vocab

    # create the embeddings from the model
    embeddings = np.zeros((nbFullWindows, embDim))
    # for all the window, compute the "curve"
    for i in range(nbFullWindows):
        for word in wordfreq.tokenize(sliding_windows[i], 'en'):
            # compute the mean of word embeddings
            if word in gloveDic.keys():
                if scale:
                    embeddings[i] += gloveDic[word]/zipf_frequency(word, 'en')
                else:
                    embeddings[i] += gloveDic[word]
            elif meanVec:
                nonVocGlove += 1
                if scale and (zipf_frequency(word, 'en') != 0):
                    embeddings[i] += meanVec/zipf_frequency(word, 'en')
                else:
                    embeddings[i] += meanVec
            else:
                nonVocGlove += 1



    embeddings /= windowSize # TODO should we count words not in the vocabulary ?
    print("Amount of words not in GLoVe vocabulary : ", nonVocGlove)

    return embeddings

def dataset_curve(texts: list, windowSize: int):
    """
    From a list of texts returns every text's curve in a list.
    """
    from tqdm import tqdm
    curve_list = []
    for i in tqdm(range(len(texts)), desc="Creating text curves..."):
        curve_list += [txt2emb(texts[i], windowSize)]

    return curve_list

def dataset_vectorized_curve(texts: list, windowSize: int, nbEmbs: int):
    """
    From a list of texts gets every text's curve and concatenate its embs together.
    Crops the curve to `nbEmbs` embeddings.
    Returns a list of vectors of the same length, for training purpose.
    """
    from tqdm import tqdm
    text_vec_list = np.zeros((len(texts), 384*nbEmbs))
    for i in tqdm(range(len(texts)), desc="Creating text curves..."):
        embs = txt2emb(texts[i], windowSize, False)
        for j in range(min(len(embs), nbEmbs)):
            text_vec_list[i][384*j:384*(j+1)] += embs[j]

    return text_vec_list

def cosine_2by2(embeddings):
    """
    builds a list detailing the 'evolution' of the sementic similarity
    ie the cosine score of the embeddings of two neighbor windows.
    """
    cos_score = []
    for i in range(len(embeddings) - 1):
        cos_score.append(cosSim(embeddings[i], embeddings[i+1]))
    return cos_score        

def cosine_fromTheBack(embeddings):
    """
    builds a list detailing the 'evolution' of the sementic similarity
    ie the cosine score of the embeddings of the beginning window vs the others.
    """
    cos_score = []
    for i in range(2,len(embeddings)):
        cos_score.append(cosSim((embeddings[0] + embeddings[1])/2, embeddings[i]))
    return cos_score        

def compute_speed(curve, windowSize: int):
    nd_embs = np.array(curve)
    speed = []
    for i in range(len(curve)-1):
        speed += [np.linalg.norm(curve[i] - curve[i+1])/windowSize]
    return speed

def plot_speeds(texts, windowSize):
    """
        Given a pandas Dataframe of labeled text `texts`, feat 2 col 'text', 'fake',
        plots the 'speed evolution' of 50 of them on the same graph, differentiated w/ color
        blue = human, orange = fake.
    """

    # query on the db 50 random rows : 25 fake and 25 true, and reset the indexing
    import random
    text_selected = pd.concat(
            [
                texts.iloc[random.sample(list(texts.query('fake == 1').index), 25)],
                texts.iloc[random.sample(list(texts.query('fake == 0').index), 25)]
            ],
            ignore_index=True)
#    rand_ids = random.sample(range(len(texts)), 50)
#    text_selected = texts.iloc(rand_ids)
    curve_list = dataset_curve(list(text_selected['text']), windowSize)

    fig, ax = plt.subplots()
    for i in range(len(curve_list)):
        speed_ev = compute_speed(curve_list[i], windowSize)
        formatting = '-C0' if text_selected['fake'][i] == 0 else '-C1'
        ax.plot(speed_ev, formatting)
    ax.set_title("Speed Evolution of 50 Human & Generated Texts", fontsize=24)
    ax.set_xlabel("Number of points in the Curve", fontsize=16)
    ax.set_ylabel("Speed", fontsize=16)
    ax.set_xlim(0, 20)
    import matplotlib.patches as mpat
    fake = mpat.Patch(color='C1', label='Generated Text')
    true = mpat.Patch(color='C0', label='Human Text')
    ax.legend(handles=[fake, true])
    plt.show()

def plot_mean_dist_curv(texts, windowSize):
    dist_human = []
    curv_human = []
    dist_generated = []
    curv_generated = []

    # Compute texts curve
    curve_list_generated = dataset_curve(list(texts.query('fake == 1')["text"]), windowSize)
    curve_list_human = dataset_curve(list(texts.query('fake == 0')["text"]), windowSize)

    # Compute mean dist and curv for each curves
    for curve in curve_list_generated:
        curv_generated.append(0); dist_generated.append(0)
        curv_generated[-1], dist_generated[-1] = characterizers(curve)
    for curve in curve_list_human:
        curv_human.append(0); dist_human.append(0)
        curv_human[-1], dist_human[-1] = characterizers(curve)

    # Compute stats on the lists.
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.boxplot([dist_human, dist_generated], labels=["Human", "Generated"])
    ax1.set_title("Mean Distance", fontsize=14)
    ax2.boxplot([curv_human, curv_generated], labels=["Human", "Generated"])
    ax2.set_title("Mean Curvature", fontsize=14)
    fig.suptitle("Comparison of Simple Curve Features", fontsize=24)
    plt.show()

def projection_and_plot(embeddings, title="", show = False):
    """
    Project the embeddings in 2d and 3d using umap and plot it
    """

    import umap
    import umap.plot
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

def characterizers(curve):
    if len(curve) < 3:
        print("Text not long enough.")
    else :
        sumAngle = 0
        sumDist = 0
        for i in range(len(curve) - 2):
            P1 = np.array(curve[i])
            P2 = np.array(curve[i + 1])
            P3 = np.array(curve[i + 2])

            sumAngle += np.arccos( np.dot(P2 - P1, P3 - P2) / np.linalg.norm(P1 - P2)/np.linalg.norm(P3 - P2) )
            sumDist += np.linalg.norm(P2- P1)

        meanAngle = sumAngle / (len(curve) - 2)
        meanDistance = (sumDist + np.linalg.norm(P3-P2)) / (len(curve) - 1)

        return meanAngle, meanDistance

if __name__ == "__main__":
    """ 
    Takes a text as first input to create its projected stylistic curve in 2d and 3d
    """
    with open(sys.argv[1]) as f:
       text = f.read()

    print("computing embeddings ...")
    embeddings = txt2emb(text, 4, False)

    
    MeanCurv, MeanDistance = characterizers(embeddings)
    print("Mean Distance : ", MeanDistance, "MeanCurvature (rad) : ", MeanCurv)
    ## Plotting of 2d and 3d curves
    title =""
    if len(sys.argv) == 3:
        title = sys.argv[2]
    projection_and_plot(embeddings, title, True)

    ## Plotting speed
    
    #speed = compute_speed(embeddings, 4)
    #plt.plot(speed)
    #plt.show()
