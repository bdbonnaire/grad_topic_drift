topWords = ["the", "be", "of", "and", "a", "in", "to", "have", "it", "for", "that", "on", "with", "do", "at", "by", "not", "this", "but", "from", "that", "or", "which", "as", "we", "an", "will"]

with open("attention_intro.txt", "r") as f:
    text = f.read().split()

cleantxt = []
for words in text:
    if words.lower() not in topWords:
        cleantxt.append(words)

txt = " ".join(cleantxt)
with open("attention_intro.clean.txt", "w") as f:

   f.write(txt) 
