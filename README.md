# Fake News Detection Using Naive Bayes

#### This repository contains an implementation of the following variants of Naive Bayes classifier for detecting fake news.  

* Naive Bayes raw (without TF-IDF)
* Naive Bayes with **TF-IDF** (Term Frequency-Inverse Document Frequency)

_Several preprocessing techniques such as **tokenization**, **stopword removal** and methods such as **Laplace smoothing** and assigning **TF-IDF** weights to probabilities, has been done to maximize the model accuracy._

_**Note**: This implementation is completely done from scratch and uses libraries for text preprocessing and evaluation purposes only._

## Dataset
The dataset used in this project is taken from [here](https://huggingface.co/datasets/ErfanMoosaviMonazzah/fake-news-detection-dataset-English).

The columns used for training are:

* **text**: The content of the news article.
* **label**: The target label (fake or real).

You can modify the script to work with your dataset by ensuring the column names match the expected structure.


## Prerequisites
* Python (latest)
* pandas
* scikit-learn
* numpy
* matplotlib
* seaborn


## Results
Both models have been evaluated according to the specified dataset only and the following results were achieved:


![Picture1](https://github.com/user-attachments/assets/661334fb-f611-4c99-8da2-5544644442fb)

![Screenshot 2024-09-30 215430](https://github.com/user-attachments/assets/6e52f40b-3d53-4896-9d66-0be5d4e8fb3b)


![Picture2](https://github.com/user-attachments/assets/88996925-0f01-4128-816d-6793b770b6a6)

![Screenshot 2024-09-30 215528](https://github.com/user-attachments/assets/50a35a09-c7ff-4178-98aa-e3c074b2ac2b)

* Accuracy (Raw Naive Bayes): 96%
* Accuracy (TF-IDF Naive Bayes): 97%

Feel free to experiment with the models and improve their performance!

### Suggested Improvements
* Word Stemming
