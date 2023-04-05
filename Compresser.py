from transformers import pipeline, BertTokenizer, BertForMaskedLM
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from itertools import combinations
import networkx as nx
import nltk
import re
nltk.download('punkt')
nltk.download('stopwords')

class Compresser:
    def __init__(self, text):
        self.text = text
        self.mask_token = '^*' # this is the token that will be replaced by the model
        '''
        self.summary = self.summarize()
        self.words = self.tokenize()
        self.to_mask = self.get_keywords()
        '''
        self.words_ = self.tokenize()
        self.key_words_ = self.get_keywords()
        self.masked_text_ = self.mask()
        self.unmasked_choices_ = self.unmask()
        self.reconstructed_text_ = self.reconstruct()

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
        threshold = 0.5 # adjust this threshold as desired
        to_mask = [word for word, score in scores.items() if score < threshold]
        return to_mask
    
    def mask(self):
        masked_text = self.text
        words = masked_text.split()
        for word in words:
            if word not in self.key_words_:
                masked_text = re.sub(r'\b{}\b'.format(re.escape(word)), self.mask_token, masked_text)
        return masked_text
    
    def unmask(self):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        unmasker = pipeline('fill-mask', model=model, tokenizer=tokenizer)
        masked_text = self.masked_text_.replace(self.mask_token, "[MASK]") #this is the masked text with ^* replaced by [MASK]
        unmasked_text = unmasker(masked_text)
        return unmasked_text
    
    def reconstruct(self):
        reconstructed_text = self.masked_text_.replace(self.mask_token, "[MASK]")
        for result in self.unmasked_choices_:
            token_str = result[0]['token_str']
            token_str = token_str.replace('[CLS]', '').replace('[SEP]', '')
            reconstructed_text = reconstructed_text.replace("[MASK]", token_str, 1)
        reconstructed_text = reconstructed_text.replace("[MASK]", self.mask_token)
        return reconstructed_text

#Just testing rq
text = "The quick brown fox jumps over the lazy dog to eat the dog's food. The dog is now hungry and wants to eat the fox. The fox is now dead. The end."
compresser = Compresser(text) #the compresser injects the text to operate on
words = compresser.key_words_ #these are the key words
masked = compresser.masked_text_ #this is the masked text with ^* instead of the words
unmasked_choices = compresser.unmask() #this is the unmasked text with the ^* replaced by the words but its a bunch of choices
print(masked)
print(words)
print(compresser.reconstructed_text_)

