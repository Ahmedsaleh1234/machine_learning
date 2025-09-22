from collections import Counter
import numpy as np

def ngram(sentence, references, n):
    #precession
    c = len(sentence)
    def clib(sentence, references, n):
        sent_clib = Counter(tuple(sentence[i: i+n]) for i in range(len(sentence)-n+1))
        max_clib_ref = Counter()
        for ref in references:
            clib_ref = Counter(tuple(ref[i:i+n]) for i in range(len(ref)-n+1))
            for w, num in clib_ref.items():
                if num > max_clib_ref[w]:
                    max_clib_ref[w] = num
        clib = sum(min(sent_clib[w], max_clib_ref.get(w, 0)) for w in sent_clib)
        return clib
    clib_sentence = Counter(tuple(sentence[i:i+n]) for i in range(len(sentence)-n+1))
    preccession = clib(sentence, references, n) / float((max(sum(clib_sentence.values()), 1)))
    #panalty
    c = len(sentence)
    close_reference_idx = np.argmin([abs(len(ref) - c) for ref in references])
    r = len(references[close_reference_idx])
    if c > r:
        p = 1
    else:
        p = np.exp(1 - r/c)
    score = p * preccession
    return score

