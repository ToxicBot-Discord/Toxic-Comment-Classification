# Toxic-Comment-Classification

- Competition : https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/
- Discord Bot Based on this : https://github.com/Sid200026/ToxicBot

The repo contains notebooks for the Jigsaw Toxic Comment Classification contest hosted on Kaggle.

Given a sentence, classify it among the following labels

- toxic
- severe_toxic
- obscene
- threat
- insult
- identity_hate

### Using Term Frequency Inverse Document Frequency

- IPYNB File : [TFIDF and Logistic Regression](ML/Toxic_Comment_Classifier_tfidf.ipynb)
- [Colab](https://colab.research.google.com/drive/1dRvXLOSmEwfRRIctLiTROt4-UVGxbXtk?usp=sharing)

#### Result

<img src="https://github.com/Sid200026/Toxic-Comment-Classification/blob/main/Output/Logistic%20Regression%20using%20TFIDF.png" alt="Output"/>

#### Steps

- Preprocessing

  - Convert to lowercase
  - Remove abbreviation, \n, double space, numeric, special characters. For example wont or won't will be converted into will not.
  - Remove url
  - Remove punctuations
  - Remove stop words like he, she, the, a etc
  - Lemmatization using WORDNET and POS Tagging

- Term Frequency Inverse Document Frequency
  - The TFIDF vector was trained on both the test and train data. The test and train dataset both consists of 1,50,000 rows. As a result, it will be very difficult for the tfidf vectorizer to provide a good result on unseen data. So we train this on the test data as well. Although this is wrong since test data shouldn't be used and it results in addition of bias, in a real world scenario the train dataset would contain enough input data so that our tfidf vectorizer can provide a good result even on unseen data.
- Logistic Regression and Multinomial Naive Bayes
  - Logistic Regression AUC-ROC : 0.98209
  - Multinomial Naive Bayers AUC-ROC : 0.95458
  - The parameters for the above two algorithms was tuned using GridSearchCV

#### Issues

- Since the dataset is large and running tfidf on a large dataset containing lot of text results in a huge datatable with numerous features. As a result it is very easy to run out of memory and face an error. I would recommend the usage of Google Colab or Kaggle Notebook to run the ipynb files.

### Using Custom-Trained Embeddings

- IPYNB File : [Custom-Trained Embedding](Toxic_Comment_Classification_Custom_Word_Embedding.ipynb)
- [Colab](https://colab.research.google.com/drive/1qUfcpwVGL3Vg0GNG9RSzE3Zd45fVKA47?usp=sharing)

#### Result

<img src="https://github.com/Sid200026/Toxic-Comment-Classification/blob/main/Output/Custom%20Word%20Embedding.png" alt="Output"/>

#### Steps

- Preprocessing

  - Use Keras Tokenizer API to convert to lowercase, split the words according to tokens and remove punctuations

- Custom Word Embeddings using a Recurrent Neural Network ( LSTM ) trained on a GPU.
- AUC-ROC : 0.97

#### Issues

- The reason for a lower score as compared to logistic regression is the fact that the custom model used texts from the train set only. So the vocabulary size was small. As a result the model failed to provide an accurate embedding for words in the test set and thus resulting in a low score.

---

### Using GloVe Embeddings

- IPYNB File : [Custom-Trained Embedding](Toxic_Comment_Classification_using_Pre_Trained_Word_Embeddings.ipynb)
- [Colab](https://colab.research.google.com/drive/1N6y43z2ioQp0fMYrRlDfnrLdtSQIDQk0?usp=sharing)
- Glove : https://nlp.stanford.edu/projects/glove/

#### Result

<img src="https://github.com/Sid200026/Toxic-Comment-Classification/blob/main/Output/GloVe%20Embedding.png" alt="Output"/>

#### Steps

- Preprocessing

  - Use Keras Tokenizer API to convert to lowercase, split the words according to tokens and remove punctuations

- GloVe Word Embeddings using a Recurrent Neural Network ( LSTM ) trained on a GPU.
- GloVe is an unsupervised learning algorithm for obtaining vector representations for words. Training is performed on aggregated global word-word co-occurrence statistics from a corpus, and the resulting representations showcase interesting linear substructures of the word vector space.
- GloVe consists of Common Crawl (840B tokens, 2.2M vocab, cased, 300d vectors, 2.03 GB download)
- AUC-ROC : 0.982

---
