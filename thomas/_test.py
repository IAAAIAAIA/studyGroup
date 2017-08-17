import nltk
# nltk.download()
from nltk.corpus import wordnet as wn

word = wn.synset('chicken.n.01')
hyper = lambda s: s.hypernyms()
print (word)
word_hyper = list(word.closure(hyper))
for i in enumerate(word_hyper):
print(i)
