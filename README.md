[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/mW4WPbr-)
# A06 - Deep Feelings

This is an optional assignment for AI (Spring 2026). See assignment details on Canvas.
## Results

| Model | Accuracy |
|---|---:|
| Baseline bag-of-words | 0.6572 |
| Stopwords removed | 0.6507 |
| Bigrams | 0.6614 |
| Stopwords + bigrams | 0.6626 |
| spaCy embeddings | 0.5832 |

## Analysis

The baseline bag-of-words model reached an accuracy of 0.6572, which is close to the expected baseline performance for this assignment. The stopword removal model did not improve performance and decreased accuracy slightly to 0.6507. This could be because some common words still give useful context in short texts like tweets, and since tweets are already brief, removing words can sometimes remove information that helps the classifier.

The bigram model improved accuracy slightly from 0.6572 to 0.6614. This suggests that two-word phrases helped capture more context than individual words alone. The stopwords + bigrams model performed best with an accuracy of 0.6626, which is an improvement of 0.0054, or 0.54 percentage points, over the baseline.

The spaCy embedding model did not improve performance and had an accuracy of 0.5832. One possible reason for this is that document vectors may lose some of the specific word-level details that are useful for sentiment classification. Since the data contains short tweets, things like exact words, slang, and phrases may be more useful than general semantic embeddings.

Overall, the best-performing enhancement was the combination of stopword removal and bigram features. However, the improvement was small, so it could partially be due to model variation or the way the dataset is split. Performance could be improved further by trying TF-IDF features, tuning Logistic Regression parameters, cleaning URLs/usernames, or using a model that is designed specifically for social media text.

## LLM Usage
The LLM Claude was used to make my code more pythonic with these prompts:
- “How can I make this scikit-learn NLP pipeline more Pythonic and readable?”
- “How can I improve the structure and commenting of this Python NLP classification code?”