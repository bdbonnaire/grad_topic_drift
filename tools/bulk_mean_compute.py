import sys
import curve_building as cb
import json

"""
first arg is the jsonl file containing the texts we want to compute the characterizers of.
Each line must have a 'text' argument.
"""

with open(sys.argv[1], 'r') as f:
    jsonList = list(f)

results = {}
outHolder = ""
for line in jsonList:
    textjson = json.loads(line)
    txtEmb = cb.txt2emb(textjson['text'], 4, False)
    results['MeanCurv'], results['MeanDist'] = cb.characterizers(txtEmb)
    outHolder += json.dumps(results) + "\n"
    
with open(sys.argv[1][:-6] + ".char.jsonl", 'w') as o:
    o.write(outHolder)
