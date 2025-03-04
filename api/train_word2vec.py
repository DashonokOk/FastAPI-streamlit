from gensim.models import Word2Vec
import nltk
from nltk.corpus import brown

nltk.download('brown')
sentences = brown.sents()

model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
model.save("weights/word2vec.model")