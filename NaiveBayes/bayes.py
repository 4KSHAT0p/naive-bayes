from math import log
from collections import Counter
import pandas as pd


# apply bayes theorem for predicting
def predict_raw(
    text: str,
    real: Counter,
    fake: Counter,
    prob_real: float,
    tot_words_real: int,
    tot_words_fake: int,
    n_unique: int,
) -> bool:
    # prob(real|words)=prob(words|real)*prob(real)/prob(words)
    prob_r = log(prob_real)
    prob_f = log(1 - prob_real)
    list_of_words = text.split()
    for word in list_of_words:
        # laplace smoothing
        prob_r += log((1 + real[word]) / (tot_words_real + n_unique))
        prob_f += log((1 + fake[word]) / (tot_words_fake + n_unique))

    return prob_r > prob_f


def IDF(df: pd.DataFrame) -> Counter:
    idf = Counter()
    num_docs = len(df)  # Total number of documents

    for txt in df["text"]:
        # Split the text into words and use a set to count each word only once per document
        words_in_doc = set(txt.split())
        # This will count each word only once per document
        for word in words_in_doc:
            idf[word] += 1

    for word in idf:
        idf[word] = log(num_docs / (1 + idf[word])) + 1  # idf smooth

    return idf


def predict_tf_idf(
    text: str,
    real: Counter,
    fake: Counter,
    tot_words_real: int,
    tot_words_fake: int,
    n_unique: int,
    idf_real: Counter,
    idf_fake: Counter,
    prob_real: float,
) -> bool:
    prob_r = log(prob_real)
    prob_f = log(1 - prob_real)
    epsilon = 1e-9
    tf = Counter(text.split())
    total = tf.total()

    for term in tf.keys():
        prob_r += log(
            (tf[term] / total)
            * (idf_real[term] + epsilon)
            * ((real[term] + 1) / (tot_words_real + n_unique))
        )
        prob_f += log(
            (tf[term] / total)
            * (idf_fake[term] + epsilon)
            * ((fake[term] + 1) / (tot_words_fake + n_unique))
        )

    return prob_r > prob_f
