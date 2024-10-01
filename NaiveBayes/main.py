import pandas as pd
from preprocess import clean_text, tokenize
from bayes import predict_raw, predict_tf_idf, IDF
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,classification_report


class NB:
    def __init__(self, dataset_path):
        # Load and preprocess training and testing data
        self.train_data = pd.read_csv(dataset_path[0], delimiter="\t")
        self.train_data = self.train_data.drop(columns=["title", "subject", "date"])
        self.train_data["text"] = self.train_data["text"].apply(clean_text)

        self.test_data = pd.read_csv(dataset_path[1], delimiter="\t")
        self.test_data = self.test_data.drop(columns=["title", "subject", "date"])
        self.test_data["text"] = self.test_data["text"].apply(clean_text)

        # these constants must be global
        self.train_true = self.train_data[self.train_data["label"] == 1]
        self.train_false = self.train_data[self.train_data["label"] == 0]

        self.prob_real = len(self.train_true) / len(self.train_data)

        self.real = tokenize(self.train_data, 1)
        self.fake = tokenize(self.train_data, 0)

        self.n_unique = len(self.real.keys() | self.fake.keys())
        self.tot_words_real = sum(self.real.values())
        self.tot_words_fake = sum(self.fake.values())

        # precomputing idf hash tables
        self.idf_real = IDF(self.train_true)
        self.idf_fake = IDF(self.train_false)

    def NaiveBayes(self):
        # Make predictions
        self.test_data["predicted_label_raw"] = self.test_data["text"].apply(
            lambda x: predict_raw(
                x, self.real, self.fake, self.prob_real, self.tot_words_real, self.tot_words_fake, self.n_unique
            )
        )

        real_labels = self.test_data["label"]
        predicted_labels = self.test_data["predicted_label_raw"]

        print(classification_report(real_labels, predicted_labels))
        
        # Generate confusion matrix
        conf_matrix = confusion_matrix(real_labels, predicted_labels)

        # Create confusion matrix heatmap using seaborn
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",cbar=False)

        # Add plot labels and title
        plt.title('Confusion Matrix for Raw Naive Bayes')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')

        # Show the plot
        plt.tight_layout()
        plt.show()

    def NaiveBayesTFIDF(self):
        self.test_data["predicted_label_tfidf"] = self.test_data["text"].apply(
            lambda x: predict_tf_idf(x,self.real,self.fake,self.tot_words_real,self.tot_words_fake,self.n_unique,self.idf_real,self.idf_fake,self.prob_real)
        )

        real_labels = self.test_data["label"]
        predicted_labels = self.test_data["predicted_label_tfidf"]

        print(classification_report(real_labels, predicted_labels))
        
        # Generate confusion matrix
        conf_matrix = confusion_matrix(real_labels, predicted_labels)

        # Create confusion matrix heatmap using seaborn
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",cbar=False)

        # Add plot labels and title
        plt.title('Confusion Matrix for Naive Bayes with TF-IDF')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')

        # Show the plot
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    model = NB(["dataset/train.tsv", "dataset/test.tsv"])
    model.NaiveBayes()
    model.NaiveBayesTFIDF()
    
