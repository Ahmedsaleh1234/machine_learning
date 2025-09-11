from collections import Counter
import numpy as np
def unigram(sentenece, reference):
    #precission
    sen_len = len(sentenece)
    sen_words_count = Counter(sentenece)
    max_ref_count = Counter()
    for i in reference:
        ref_count = Counter(i)
        for word, num in ref_count.items():
            if num > max_ref_count[word]:
                max_ref_count[word] = num
    clib = sum(min(sen_words_count[word], max_ref_count.get(word, 0)) for word in sen_words_count)
    preccission = clib / sen_len
    ref_close_len_idx = np.argmin([abs(len(x) - sen_len) for x in reference])
    ref_close_len = len(reference[ref_close_len_idx])
    if sen_len > ref_close_len:
        p  = 1
    else:
        p = np.exp(1 - ref_close_len/sen_len)
    unigram_score = p * preccission
    return unigram_score
