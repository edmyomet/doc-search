import os
import re 
import json 
from pdfminer.high_level import extract_text
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO
import itertools

from transformers import AutoTokenizer

import nltk
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tree import Tree
from nltk import pos_tag, ne_chunk


from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd 

import pickle

nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
def init_dict_copy(**kwargs)->dict:
    copy:dict = kwargs['copy']
    return {i:[] for i in copy.keys()}

def get_all_file_data(**kwargs)->list:
    path:str = kwargs['document_folder']
    buffer = StringIO()
    layout_param = LAParams()
    resource_manager = PDFResourceManager()
    device = TextConverter(resource_manager, buffer, laparams=layout_param)
    interpreter = PDFPageInterpreter(resource_manager, device)
    
    doc_text:dict = {file_name:[] for file_name in os.listdir(path)}
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            with open(file_path, 'rb') as file:
                for page in PDFPage.get_pages(file):
                    interpreter.process_page(page)
            text = buffer.getvalue()
            doc_text[filename].append(text)
        except:
            print(f'Error Extracting: {filename}')
    
    return doc_text
def extract_named_entities(**kwargs):
    named_entities = set()
    chunked = ne_chunk(pos_tag(kwargs['tokens']))
    for subtree in chunked:
        if isinstance(subtree, Tree):
            entity = "".join([token for token, pos in subtree.leaves()])
            named_entities.add(entity.lower())
    return named_entities

class Preprocessing:
    document_data:dict = {}
    document_folder:str = ""
    document_tokenised:dict = {}
    document_lemmatized:dict = {}
    documents:dict={}
    
    _lemmatizer = None
    _tokenizer = None
    def __init__(self, *args):
        self.document_folder = args[0]
        self.document_data = get_all_file_data(document_folder=self.document_folder)
        self.document_tokenised = init_dict_copy(copy=self.document_data)
        self.document_lemmatized = init_dict_copy(copy=self.document_data)
        self.documents = init_dict_copy(copy=self.document_data)
    def __remove_special_characters(self):
        tmp_text:str
        for file_name, data in self.document_data.items():
            tmp_text = "".join(data)
            tmp_text = re.sub(r'[^a-z\s]', '', tmp_text)
            # tmp_text = re.sub(r'[,\':;!@#$%&*()\'\'\"\"]+', "", tmp_text)
            # tmp_text = re.sub(r"\s+", "", tmp_text)
            # tmp_text = re.sub(r"\n+", "", tmp_text)
            self.document_data[file_name] = tmp_text 
    
    def __tokenize(self):
        #self._tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        for file_name, data in self.document_data.items():
            self.document_tokenised[file_name] = word_tokenize(data)
    
    def __lemmatize(self):
        self._lemmatizer = WordNetLemmatizer()
        for file_name, data in self.document_tokenised.items():
            for word, tag in pos_tag(data):
                pos = tag[0].lower()
                pos = pos if pos in ['a', 'r', 'n', 'v'] else 'n'
                self.document_lemmatized[file_name].append(self._lemmatizer.lemmatize(word, pos))
                
        
    def __stop_word_removal(self):
        stop_words = set(stopwords.words('english'))
        for file_name, data in self.document_lemmatized.items():
            named_entities = extract_named_entities(tokens=data)
            self.documents[file_name] = [token for token in data if token not in stop_words or token in named_entities or token.isalpha()]                   

    
    def preprocess(self):
        self.__remove_special_characters()
        self.__tokenize()
        self.__lemmatize()
        self.__stop_word_removal()
    
    def get(self):
        self.preprocess()
        return self.documents

class NLP(Preprocessing):    
    vectorizer = TfidfVectorizer(ngram_range=(1, 2),stop_words='english',sublinear_tf=True, norm='l2',
                                 smooth_idf=True, use_idf=True)
    doc_names:list = []
    doc_content:list = []
    tf_idf_matrix = None
    tf_idf_df = None
    def __init__(self, *args):
        super().__init__(args[0])
        self.preprocess()
    
    def __convert_to_dataframe(self):
        
        self.tf_idf_df = pd.DataFrame(
            self.tf_idf_matrix.toarray(), 
            index=self.doc_names, 
            columns= self.vectorizer.get_feature_names_out()
        )
    def tfidf_matrix(self):
        self.doc_content = [" ".join(self.documents[doc]) for doc in self.documents]
        #self.doc_names = [f'doc-{i+1}' for i in range(len(self.doc_content))]
        self.doc_names = list(self.document_data.keys())
        self.tf_idf_matrix = self.vectorizer.fit_transform(self.doc_content)
        self.__convert_to_dataframe()
    
    def save_dataframe(self):
        self.tf_idf_df.to_pickle(rf'results/TF-IDF.pkl')
        

        
    
def main():
    nlp = NLP(rf'documents/')
    nlp.tfidf_matrix()
    nlp.save_dataframe()


if __name__ == '__main__':
    main()
    