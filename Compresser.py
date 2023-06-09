from transformers import pipeline, BertTokenizer, BertForMaskedLM
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from itertools import combinations
import networkx as nx
import nltk
import re
nltk.download('punkt')
nltk.download('stopwords')


class Compresser:
    '''
    This class compresses a text by removing redundant information.
    '''
    def __init__(self, text):
        '''
        text: the text to be compressed
        attributes:
            text: the text to be compressed
            mask_token: the token that will be replaced by the model
            summary_: the summary of the text
            words_: the words in the text
            key_words_: the key words in the text
            masked_text_: the masked text
            unmasked_choices_: the unmasked choices
            reconstructed_text_: the reconstructed text
        methods:
            summarize: summarizes the text
            tokenize: tokenizes the text
            get_keywords: gets the keywords in the text by using PageRank
            mask: masks the text
            unmask: unmask the masked text by using BERT
            reconstruct: reconstructs the text by replacing the masked tokens with the unmasked tokens
        '''
        self.text = text
        self.mask_token = '^*' # this is the token that will be replaced by the model
        #Gens
        #self.summary_ = self.summarize()
        self.words_ = self.tokenize()
        self.key_words_ = self.get_keywords()
        self.masked_text_ = self.mask()
        self.unmasked_choices_ = self.unmask()
        self.reconstructed_text_ = self.reconstruct()

    def summarize(self):
        '''
        Summarizes the text.
        Returns:
            summary: The summary of the text.
        '''
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        summary = summarizer(self.text, max_length=1000, min_length=3, do_sample=False)
        return summary[0]['summary_text']
    
    def tokenize(self):
        '''
        Tokenizes the text into sentences and words.
        Returns:
            words: The tokenized words, grouped by sentence. 
            Each inner list represents a sentence and contains the individual
            words as strings.
        '''
        # Tokenize text into sentences and words
        sentences = sent_tokenize(self.text)
        words = [word_tokenize(sentence.lower()) for sentence in sentences]
        # Remove stop words and punctuation
        stop_words = set(stopwords.words('english'))
        words = [[word for word in sentence if word.isalnum() and word not in stop_words] for sentence in words]
        return words

    def get_keywords(self):
        '''
        Gets the keywords in the text by using PageRank.
        Returns:
            to_mask: The keywords in the text.
        '''
        graph = nx.Graph()
        for sentence in self.words_:
            for u, v in combinations(sentence, 2):
                if not graph.has_edge(u, v):
                    graph.add_edge(u, v, weight=1)
                else:
                    graph.edges[u, v]['weight'] += 1
        scores = nx.pagerank(graph, weight='weight')
        threshold = 0.05 # adjust this threshold as desired
        to_mask = [word for word, score in scores.items() if score < threshold]
        return to_mask
    
    def mask(self):
        '''
        Masks the text.
        Returns:
            masked_text: The masked text.
        '''
        masked_text = self.text
        words = masked_text.split()
        for word in words:
            if word not in self.key_words_:
                masked_text = re.sub(r'\b{}\b'.format(re.escape(word)), self.mask_token, masked_text)
        return masked_text
    
    def unmask(self):
        '''
        Unmask the masked text by using BERT.
        Returns:
            unmasked_choices: The unmasked choices. The format is a JSON object represetning potential choices for each masked token.
        '''
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        unmasker = pipeline('fill-mask', model=model, tokenizer=tokenizer)
        masked_text = self.masked_text_.replace(self.mask_token, "[MASK]") #this is the masked text with ^* replaced by [MASK]
        unmasked_text = unmasker(masked_text)
        return unmasked_text

    def reconstruct(self):
        '''
        Reconstructs the text by replacing the masked tokens with the unmasked tokens.
        This is done by replacing the list of unmasked choices by filling in the highest scoring
        choice for each masked token.
        To Do:
            - Make this more robust by using the sentence embeddings to determine which choice is the best
        Returns:
            reconstructed_text: The reconstructed text.
        '''
        reconstructed_text = self.masked_text_.replace(self.mask_token, "[MASK]")
        for result in self.unmasked_choices_:
            token_str = result[0]['token_str']
            token_str = token_str.replace('[CLS]', '').replace('[SEP]', '')
            reconstructed_text = reconstructed_text.replace("[MASK]", token_str, 1)
        reconstructed_text = reconstructed_text.replace("[MASK]", self.mask_token)
        return reconstructed_text