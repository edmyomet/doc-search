from NLP import *
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from sklearn.metrics import jaccard_score
import numpy as np
nlp = NLP(rf'documents/')
nlp.tfidf_matrix()
document_matrix = nlp.get()
from scipy.spatial.distance import correlation

def preprocess(**kwargs):
    query = kwargs['query']
    query = re.sub(r'[^a-z\s]', '', query)
    
    tokens = word_tokenize(query.lower())
    
    lemmatized:list = []
    for word, tag in pos_tag(tokens):
        pos = tag[0].lower()
        pos = pos if pos in ['a', 'r', 'n', 'v'] else 'n'
        lemmatized.append(nlp._lemmatizer.lemmatize(word, pos))
    
    stop_words = set(stopwords.words('english'))
    named_entities = extract_named_entities(tokens=tokens)
    final_tokens = [token for token in lemmatized if token not in stop_words or token in named_entities or token.isalpha()]
    
    return final_tokens

def search_documents_pearson(query):
    processed_query = preprocess(query=query)
    query_str = " ".join(processed_query)
    query_vector = nlp.vectorizer.transform([query_str])
    
    
    pearson_similarities = []
    for doc_vector in nlp.tf_idf_matrix:
        similarity = 1 - correlation(query_vector.toarray()[0], doc_vector.toarray()[0])
        pearson_similarities.append(similarity)
    
    
    ranked_indices = np.argsort(pearson_similarities)[::-1]
    ranked_results = [(nlp.doc_names[i], pearson_similarities[i]) for i in ranked_indices]
    
    return ranked_results



def search_results(**kwargs):
    query = kwargs['query']
    choice = kwargs['distance']
    processed_query = preprocess(query=query)
    query_vector = nlp.vectorizer.transform(["".join(processed_query)])
    match(choice):
        case 'EU':
            similarity_scores = euclidean_distances(query_vector, nlp.tf_idf_matrix)
        case 'MH':
            similarity_scores = manhattan_distances(query_vector, nlp.tf_idf_matrix)
        case 'CS':
            similarity_scores = cosine_similarity(query_vector, nlp.tf_idf_matrix)
    
    ranked_indices = similarity_scores[0].argsort()
    ranked_results = [(nlp.doc_names[i], similarity_scores[0][i]) for i in ranked_indices]
    return ranked_results

def search_documents_jaccard(query):
    processed_query = preprocess(query=query)
    query_str = " ".join(processed_query)
    query_vector = nlp.vectorizer.transform([query_str])
    
    
    jaccard_similarities = []
    for doc_vector in nlp.tf_idf_matrix:
        
        query_bin = (query_vector > 0).astype(int).toarray()
        doc_bin = (doc_vector > 0).astype(int).toarray()
        
        
        jaccard_sim = jaccard_score(query_bin[0], doc_bin[0])
        jaccard_similarities.append(jaccard_sim)
    
    
    ranked_indices = np.argsort(jaccard_similarities)[::-1]
    ranked_results = [(nlp.doc_names[i], jaccard_similarities[i]) for i in ranked_indices]
    
    return ranked_results
    
def search_documents_dot_product(query):
    processed_query = preprocess(query=query)
    query_str = " ".join(processed_query)
    query_vector = nlp.vectorizer.transform([query_str])
    
    
    dot_products = np.array([query_vector.dot(doc_vector.T).toarray()[0][0] for doc_vector in nlp.tf_idf_matrix])
    
    
    ranked_indices = dot_products.argsort()[::-1]
    ranked_results = [(nlp.doc_names[i], dot_products[i]) for i in ranked_indices]
    
    return ranked_results


def main():
    while True:
        query = input("Enter any search query: ")
        if query.lower() == "stop":
            break
        result_set = [search_documents_jaccard(query), search_results(query=query, distance='CS'),
                      search_results(query=query, distance='MH'), search_results(query=query, distance='EU'), 
                      search_documents_pearson( query), search_documents_dot_product(query)]
    
        for result in result_set:
            for doc_name, score in result[:4]:
                print(f'Document:{doc_name}, Similarity_Score/Distance:{score}')
            print()
            print()
        

if __name__ == '__main__':
    main()
        