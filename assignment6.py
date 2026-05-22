import pandas as pd
import numpy as np
import spacy

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline


TRAIN_FILE = "A06_train.csv"
TEST_FILE = "A06_test.csv"


def load_data():
    """
    Loads the training and testing CSV files.

    The files do not appear to have header rows, so header=None tells
    pandas to treat every row as actual data instead of column names.
    """
    train_df = pd.read_csv(TRAIN_FILE, header=None)
    test_df = pd.read_csv(TEST_FILE, header=None)

    train_df.columns = ["sentiment", "text"]
    test_df.columns = ["sentiment", "text"]

    x_train = train_df["text"].astype(str)
    y_train = train_df["sentiment"]

    x_test = test_df["text"].astype(str)
    y_test = test_df["sentiment"]

    return x_train, y_train, x_test, y_test


def evaluate_model(name, model, x_train, y_train, x_test, y_test):
    """
    Trains one model, tests it, and prints accuracy/results.
    """
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, predictions))

    return accuracy


def get_spacy_vectors(texts, nlp):
    """
    Faster spaCy vector generation using nlp.pipe().
    """

    vectors = []

    for doc in nlp.pipe(texts, batch_size=64):
        vectors.append(doc.vector)

    return np.array(vectors)


def main():
    x_train, y_train, x_test, y_test = load_data()

    results = {}

    # Baseline system:
    # Uses bag-of-words features with CountVectorizer.
    baseline_model = Pipeline([
        ("vectorizer", CountVectorizer()),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    # Enhanced system 1:
    # Removes common English stopwords like "the", "and", and "is".
    stopwords_model = Pipeline([
        ("vectorizer", CountVectorizer(stop_words="english")),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    # Enhanced system 2:
    # Uses both single words and two-word phrases as features.
    bigram_model = Pipeline([
        ("vectorizer", CountVectorizer(ngram_range=(1, 2))),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    # Additional enhanced system:
    # Combines stopword removal with unigram and bigram features.
    stopwords_bigram_model = Pipeline([
        ("vectorizer", CountVectorizer(stop_words="english", ngram_range=(1, 2))),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    results["Baseline bag-of-words"] = evaluate_model(
        "Baseline bag-of-words",
        baseline_model,
        x_train,
        y_train,
        x_test,
        y_test
    )

    results["Stopwords removed"] = evaluate_model(
        "Enhanced model 1: stopwords removed",
        stopwords_model,
        x_train,
        y_train,
        x_test,
        y_test
    )

    results["Bigrams"] = evaluate_model(
        "Enhanced model 2: unigram + bigram features",
        bigram_model,
        x_train,
        y_train,
        x_test,
        y_test
    )

    results["Stopwords + bigrams"] = evaluate_model(
        "Enhanced model 3: stopwords removed + bigrams",
        stopwords_bigram_model,
        x_train,
        y_train,
        x_test,
        y_test
    )

    # Embedding system:
    # Uses spaCy vectors instead of CountVectorizer features.
    print("\n" + "=" * 60)
    print("spaCy embedding model")
    print("=" * 60)

    nlp = spacy.load("en_core_web_md")

    x_train_vectors = get_spacy_vectors(x_train, nlp)
    x_test_vectors = get_spacy_vectors(x_test, nlp)

    embedding_model = LogisticRegression(max_iter=1000)
    embedding_model.fit(x_train_vectors, y_train)

    embedding_predictions = embedding_model.predict(x_test_vectors)
    embedding_accuracy = accuracy_score(y_test, embedding_predictions)

    results["spaCy embeddings"] = embedding_accuracy

    print(f"Accuracy: {embedding_accuracy:.4f}")
    print(classification_report(y_test, embedding_predictions))

    # Final summary makes it easy to compare models
    # and copy the results into the README.
    print("\n" + "=" * 60)
    print("Summary of Results")
    print("=" * 60)

    for model_name, accuracy in results.items():
        print(f"{model_name}: {accuracy:.4f}")


if __name__ == "__main__":
    main()