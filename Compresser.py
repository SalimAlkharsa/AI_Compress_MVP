from transformers import pipeline
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from itertools import combinations
import networkx as nx
import nltk
nltk.download('punkt')
nltk.download('stopwords')

class Compresser:
    def __init__(self, text):
        self.text = text
        self.mask_token = '[MASK]' # this is the token that will be replaced by the model
        '''
        self.summary = self.summarize()
        self.words = self.tokenize()
        self.to_mask = self.get_keywords()
        '''
        self.words_ = self.tokenize()
        self.key_words_ = self.get_keywords()

    def summarize(self):
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        summary = summarizer(self.text, max_length=100, min_length=3, do_sample=False)
        return summary[0]['summary_text']
    
    def tokenize(self):
        # Tokenize text into sentences and words
        sentences = sent_tokenize(self.text)
        words = [word_tokenize(sentence.lower()) for sentence in sentences]
        # Remove stop words and punctuation
        stop_words = set(stopwords.words('english'))
        words = [[word for word in sentence if word.isalnum() and word not in stop_words] for sentence in words]
        return words

    def get_keywords(self):
        graph = nx.Graph()
        for sentence in self.words_:
            for u, v in combinations(sentence, 2):
                if not graph.has_edge(u, v):
                    graph.add_edge(u, v, weight=1)
                else:
                    graph.edges[u, v]['weight'] += 1
        scores = nx.pagerank(graph, weight='weight')
        threshold = 0.1 # adjust this threshold as desired
        to_mask = [word for word, score in scores.items() if score < threshold]
        return to_mask

#Just testing rq
text = "The quick brown fox jumps over the lazy dog to eat the dog's food. The dog is now hungry and wants to eat the fox. The fox is now dead. The end."
compresser = Compresser(text)
words = compresser.key_words_
print(words)