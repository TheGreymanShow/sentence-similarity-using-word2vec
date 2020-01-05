import numpy as np
from gensim.models import Word2Vec


def load_word2vec_model():
    """
    Load pretrained model (since intermediate data is not included, the model cannot be refined with additional data)
    :return: The pertrained model that is loaded.
    """
    model = Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, norm_only=True)
    return model


def get_aggregate_vector(vectors):
    """
    Form an average vetor from a list of vectors
    :param vectors: the list of vectors of words in a sentence/document
    :return: the aggregate vector
    """

    aggregate_vector = np.zeros((300, 1))
    for vec in vectors:
        vec = np.array(vec).reshape(300, 1)
        aggregate_vector += vec

    aggregate_vector = aggregate_vector / len(vectors)
    return aggregate_vector


def get_vector(word, model):
    """
    Function to fetch the vector for a word the model of pre-trained vectors.
    """
    return model.wv[word]


def vectorize_sentence(sentence, model):
    """
    Function to aggregate all the word vectors to form a sentence vector
    :param sentence: List of word vectors
    :param model: ConoceptNet Embedding model
    :return: A aggregated sentence vector
    """
    final_vec = np.zeros(300, )
    count = 0
    for word in sentence:
        count += 1
        dummy_vec = np.zeros(300, )
        try:
            temp_vec = get_vector(word, model)
            final_vec += temp_vec
        except:
            final_vec += dummy_vec
    return final_vec / count


def vectorize_documents(documents, model):
    """
    Function to aggregate all sentence vectors to form a document vector
    :param documents: document string or textssssss
    :param model: ConceptNet model of pre-trained vectors
    :return:
    """
    document_vectors = []
    count = 0
    for document in documents:
        count += 1
        sentence_vectors = [vectorize_sentence(sentence, model) for sentence in document]
        document_vector = get_aggregate_vector(sentence_vectors)
        document_vectors.append(document_vector)
    return document_vectors


def compute_cosine_sim(vec1, vec2):
    """ Compute the cosine similarity between two vectors """
    numer = np.dot(vec1.reshape((300,)), vec2.reshape((300,)))
    denom = np.sqrt(np.sum(np.square(vec1.reshape(300, )))) * np.sqrt(
        np.sum(np.square(vec2.reshape(300, ))))

    similarity = numer / denom

    return similarity


def vectorize_sentences(sentences, model):
    sentence_vectors = [vectorize_sentence(sent, model) for sent in sentences]
    return sentence_vectors


def normalize(scores, default_score):
    """
    Normalize all cosine similarity scores in a range 0-1 with max score being scaled to 1 and lowest to 0
    :param scores: List of cosine similarity scores
    :param default_score: default dummy score in case of a division by zero error
    :return: the list of normalized scores
    """
    if len(scores) > 0:
        max_scores = max(scores)
        min_scores = min(scores)

        if max_scores == min_scores:
            return [default_score] * len(scores)
        scores = [float((score - min_scores) / (max_scores - min_scores)) for score in scores]
        return scores
    else:
        return [default_score]


# Similarity between sentences

# 1. Fill in the list of sentences
sentences = ["", "", ""]

# 2. Load the word2vec model
model = load_word2vec_model()

# 3. Generate sentence vectors using word2vec model
sent_vectors = vectorize_sentences(sentences, model)

# 4. Compute the similarity score of each sentence with every other sentence in the list
sent_similarities = []
for i, sentence_one in enumerate(sentences):
    for j, sentence_two in enumerate(sentences):
        similarity = {}
        similarity["sentence_1"] = sentence_one
        similarity["sentence_2"] = sentence_two
        similarity["similarity"] = compute_cosine_sim(sent_vectors[i], sent_vectors[j])
        sent_similarities.append(similarity)

print(sent_similarities)

# Similarity between documents

# 1. Fill in the list of documents
documents = ["", "", ""]

# 2. Load the word2vec model
model = load_word2vec_model()

# 3. Generate sentence vectors using word2vec model
sent_vectors = vectorize_documents(documents, model)

# 4. Compute the similarity score of each document with every other document in the list
doc_similarities = []
for i, document_one in enumerate(documents):
    for j, document_two in enumerate(documents):
        similarity = {}
        similarity["document_1"] = document_one
        similarity["document_2"] = document_two
        similarity["similarity"] = compute_cosine_sim(sent_vectors[i], sent_vectors[j])
        doc_similarities.append(similarity)

print(doc_similarities)