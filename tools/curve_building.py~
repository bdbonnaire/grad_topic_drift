#! /usr/bin/python

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-MiniLM-L6-v2') # TODO investigate changing the model


def txt2emb(text):
    # create a list of the words in the text
    wordedText = text.split()
    windowSize = 5
# Cuts the given text into a sliding window of `windowSize` words
    sliding_window = [" ".join(wordedText[i:i+windowSize]) for i in range(len(wordedText))]

# create the embeddings from the model
    embeddings = model.encode(sliding_window)

#save the result to a file
    return embeddings

if __name__ == "main":
# TODO do we need to clean the sentences first or is it done by the encoding ?
    with open(sys.argv[1]) as f:
       text = f.read()

    embeddings = txt2emb(text)

    with open(sys.argv[2],'w') as o:
        o.write(str(embeddings))
