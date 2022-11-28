# import spacy # time consuming
from phonemizer import phonemize
import editdistance
# from fastdtw import fastdtw
import csv
from itertools import permutations
# import matplotlib.pyplot as plt
import numpy as np
# import panphon.panphon as panphon

def lexeme_to_phoneme(sentence):
    return phonemize(sentence, backend='espeak')

# Possible extension: find 
def piecewise_distance(s, t):
    ft = panphon.FeatureTable()
    return 0 if s==t else 1

# function from https://github.com/MJeremy2017/machine-learning-models
def dtw(s, t):
    n, m = len(s), len(t)
    dtw_matrix = np.zeros((n+1, m+1))
    dtw_matrix.fill(np.inf)
    dtw_matrix[0, 0] = 0
    
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = piecewise_distance(s[i-1], t[j-1])
            # take last min from a square box
            last_min = np.min([dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]])
            dtw_matrix[i, j] = cost + last_min
    return dtw_matrix[-1][-1]


def compare_distance(sent1, sent2):
    sent1, sent2 = sent1.lower().replace(' ',''), sent2.lower().replace(' ','')

    edit_distance = editdistance.eval(sent1, sent2)/((len(sent1) + len(sent2))/2)
    dtw_distance = dtw(sent1, sent2)/((len(sent1) + len(sent2))/2)

    return edit_distance, dtw_distance

def calculate_distances(sentences, f = editdistance.eval):
    distances = [[f(sentences[i][0].replace(' ',''), sentences[j][1].replace(' ',''))/((len(sentences[i][0]) + len(sentences[j][1]))/2) for j in range(len(sentences))] for i in range(len(sentences))]
    
    return distances

'''
def draw_heatmap(arr, title):
    arr = np.array(arr)
    score = 1 - arr.diagonal().sum() / arr.sum()
    fig, ax = plt.subplots()
    im = ax.imshow(arr, cmap="magma", interpolation='nearest')
    ax.set_xlabel('Subtitle')
    ax.set_ylabel('Reconstructed')
    for i in range(len(arr)):
        for j in range(len(arr[0])):
            text = ax.text(j, i, f'{arr[i, j]:.2f}',
                        ha="center", va="center", color="k")
    ax.set_title(f'{title} (score:{score:.3f})')
    plt.savefig(f'{title}.png')
    plt.show()
'''

def match_subtitles(sentences):
    phoneme_sentences = [(lexeme_to_phoneme(s1), lexeme_to_phoneme(s2)) for (s1, s2) in sentences]
    distances = calculate_distances(phoneme_sentences)
    total_distances = []
    xs = np.arange(len(sentences))
    for i, ys in enumerate(permutations(xs)):
        total = 0.0
        for x, y in enumerate(ys):
            total += distances[x][y]
        total_distances.append(total)
    argmin = np.argmin(total_distances)
    idxs = list(permutations(xs))[argmin]
    matches = [(sentences[i][0], sentences[idxs[i]][1]) for i in range(len(sentences))]
    return dict(matches)

if __name__ == '__main__':
    # sentence = 'The DNA code used for life is near universal. All forms of life and viruses use essentially the same genetic code'.lower()
    # sentence2 = 'THE DNA COULD USE FOR LIFE IS NEARLY UNIVERSAL ALL FORMS OF LIFE AND VIRUSES USE ESSENTIALLY THE SAME GENETIC'.lower()
    # print(editdistance.eval(sentence, sentence2))
    # print(lexeme_to_phoneme(sentence))
    # print(lexeme_to_phoneme(sentence2))
    # print(editdistance.eval(lexeme_to_phoneme(sentence), lexeme_to_phoneme(sentence2)))
    with open('221113_lsmma_tsv.tsv') as fr:
        reader = csv.reader(fr, delimiter='\t')
        next(reader)
        sentences = list(map(lambda x:(x[1], x[2]), reader))
        sentences = list(filter(lambda x:x[1] != '', sentences))
        
        # phoneme_sentences = [(lexeme_to_phoneme(s1), lexeme_to_phoneme(s2)) for (s1, s2) in sentences]
        # calculate_distances(phoneme_sentences[:3])

        # lex_dists, pho_dists = [], []
        # lex_dtw, pho_dtw = [], []
        # for i in range(len(sentences)):
        #     lex_dists.append([])
        #     pho_dists.append([])
        #     lex_dtw.append([])
        #     pho_dtw.append([])
        #     for j in range(len(sentences)):
        #         l, ld = compare_distance(sentences[i][0], sentences[j][1])
        #         p, pd = compare_distance(phoneme_sentences[i][0], phoneme_sentences[j][1])
        #         lex_dists[i].append(l)
        #         pho_dists[i].append(p)
        #         lex_dtw[i].append(ld)
        #         pho_dtw[i].append(pd)

        # draw_heatmap(lex_dists, 'lexeme edit distance')
        # draw_heatmap(pho_dists, 'phoneme edit distance')
        # draw_heatmap(lex_dtw, 'lexeme dtw')
        # draw_heatmap(pho_dtw, 'phoneme dtw')

        sentences_subset = [sentences[0], sentences[1], sentences[4], sentences[7]]
        print(*match_subtitles((sentences_subset)), sep='\n')