from sys import argv
from tools.curve_building_ctxt_free_originalScaling import *

def hard_plot(ax, curv, legend=True, title:str=''):
    cos2x2 = cosine_2by2(curv)
    cosine_back = cosine_fromTheBack(curv)
    [cos2x2_line] = ax.plot(cos2x2, 'b-', label="one to one")
    [cosBack_line] = ax.plot(cosine_back, 'C0-', label="from the back")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Window Number', fontsize=14)
    ax.set_ylabel('Cosine Similarity', fontsize=14)
    # lines to represent the transition in hard drift text
    cutline = ax.axvline(x=56//windowSize +1, label="Abrupt Transition in Text", ls='--', c='red', lw=1)
    ax.axvline(x=117//windowSize +1, ls='--', c='red', lw=1)
#mathline = ax.axvline(x=4, ymax=0.5, label="Math Filled Sentence",ls='--', c='green', lw=1)
    if legend : ax.legend(handles=[cutline, cos2x2_line, cosBack_line])
    ax.set_xticks(ticks=range(0,len(cosine_back)))

with open(argv[1], 'r') as f:
        text = f.read()

windowSize = 8
curvNoMean  =  txt2emb(text,  windowSize,  isJumping=True,  meanVec=False)
curvMean    =  txt2emb(text,  windowSize,  isJumping=True,  meanVec=True)

plt.style.use('seaborn')
fig, [axnoMean, axMean] = plt.subplots(2, 1)
fig.suptitle('Cosine Similarity Evolution with GloVe Embeddings', fontsize=20)

hard_plot(axnoMean, curvNoMean, title='Non-vocabulary words are ignored')
hard_plot(axMean, curvMean, legend=False, title='Non-vocabulary words are replaced by mean vector')


plt.show()
