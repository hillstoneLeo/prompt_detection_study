# # LLM Security - Prompt Injection

# ## Part 1 - Classification Using Traditional ML

# ### 1. INITIALIZATION

import pandas as pd
from transformers import BertTokenizer, BertModel
import torch 
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ### 2. LOADING AND EXPLORING DATASET

data_train = pd.read_parquet("./data/data/train-00000-of-00001-9564e8b05b4757ab.parquet")
data_test = pd.read_parquet("./data/data/test-00000-of-00001-701d16158af87368.parquet")

# As can be inferred from the previous code, the original data set is already split into training and holdout subsets.
#
# One approach is to combine both parts into a unified collection and resort to the random static train/test split or cross validation techniques to re-produce the testing samples and evaluate methods performance.
#
# However, since we planned to perform several experiments, we decided to maintain this original split for the multiple experiments to compare results using a unified testing benchmark.

# Check training data set head
data_train.head()

# Check testing data set head
data_test.head()

# Check training data set info
data_train.info()

# As can be seen in this brief overview, the data set structure is quite simple and has two columns, one for the prompt text itself and another one indicating the label if the prompt has injection or not.

# Rename "text" column into "prompt"
data_train.rename(columns={"text":"prompt"}, inplace=True)
data_test.rename(columns={"text":"prompt"}, inplace=True)

# ### 3. TEXT EMBEDDING

# According to the dataset creators, they incorporated some samples in languages other than English to extend the attack surface of prompt injection into other languages. This also helps creating universal detection models which are language agnostic and thus more capable of defending against malicious prompts in different languages.
#
# Having that said, the traditional pre-trained language-specific Word2Vec embeddings won't fit our use case, since we need embedding models that represent text of several languages in the same vector space.
#
# For this reason, we resort to a more recent language model, namely multilingual BERT to obtain embeddings. BERT (Bidirectional Encoder Representations from Transformers) has multilingual models that are pre-trained on a vast amount of data and can be used for obtaining contextualized word embeddings for various languages.

# #### Tokenization and Embedding (Hugging Face BERT Model)

# Load pre-trained multilingual BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
model = BertModel.from_pretrained('bert-base-multilingual-uncased')


# Function to tokenize and get embeddings for each prompt text
def get_bert_embedding(prompt):
    tokens = tokenizer(prompt, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**tokens)
    last_hidden_states = outputs.last_hidden_state
    embedding_vector = last_hidden_states.mean(dim=1).squeeze().numpy()
    return embedding_vector


# Apply embedding function to prompts and create a new "embedding" column
data_train['embedding'] = data_train['prompt'].apply(get_bert_embedding)
data_test['embedding'] = data_test['prompt'].apply(get_bert_embedding)

# Check how the embeddings look like inside the dataset
data_train

# Noticing the "embedding" column, each cell contains the embedding vector representing the corresponding prompt. 
#
# The content of this column along with the class "label" would represent the input for classification algorithms after being unpacked into several independent columns.

# ### 4. MODEL TRAINING

X_train = pd.DataFrame(data_train["embedding"].to_list())
y_train = data_train["label"]
X_test = pd.DataFrame(data_test["embedding"].to_list())
y_test = data_test["label"]

print(f"#Training Samples: {len(X_train)}")
print(f"#Testing Samples: {len(X_test)}")

estimators = [
    ("Naive Bayes", GaussianNB()),
    ("Logistic Regression", LogisticRegression()),
    ("Support Vector Machine", svm.SVC()),
    ("Random Forest", RandomForestClassifier())
]


results = pd.DataFrame(columns=["accuracy", "precision", "recall", "f1 score"])

for est_name, est_obj in estimators:
    est_obj.fit(X_train, y_train)
    y_predict = est_obj.predict(X_test)
    accuracy = accuracy_score(y_test, y_predict)
    precision = precision_score(y_test, y_predict)
    recall = recall_score(y_test, y_predict)
    f1 = f1_score(y_test, y_predict)
    results.loc[est_name] = [accuracy, precision, recall, f1]

# ### 5. RESULT ANALYSIS

results

# Looking at the obtained results, we observe the following points:
#
# 1. The majority of the considered models demonstrated relatively good performance on the testing data samples.
# 2. Despite Naive Bayes having the lowest performance, it still managed to perform well, achieving a very good overall accuracy.
# 3. Logistic Regression and Support Vector Machine exhibited the best outcomes, with highly accurate predictions and impressive f1 score.
# 4. Among the four evaluated models, three attained full precision, with no False Positive predictions.
# 5. Apart from Naive Bayes, errors are concentrated in the recall metric, indicating instances where positive injection prompts were erroneously predicted as normal. This is particularly important to address in this kind of problems to prevent malicious prompts from being passed to the LLM.

# In summary, Logistic Regression and Support Vector Machine stand out as top performers across all metrics, with Logistic Regression achieving the highest overall F1 score. Naive Bayes and Random Forest, while having a lower F1 score, still demonstrates respectable performance.

# Finally, since all models failed to predict all positive samples, we will take a look into some of the injections that were incorrectly predicted as negative.
#
# For this purpose, we will check the best-performing model among all others, Logistic Regression.

model = [est[1] for est in estimators if est[0] == "Logistic Regression"][0]

# Predict test samples
y_predict = model.predict(X_test)

# Append predictions to test dataset
data_test["predicted"] = y_predict

# Retrieve a sample of correctly classified prompt injections
data_test[(data_test["label"] == data_test["predicted"]) & (data_test["label"] == 1)]["prompt"].head().tolist()

# Retrieve the misclassified prompts
data_test[data_test["label"] != data_test["predicted"]]["prompt"].tolist()

# We make the following observations:
# 1. Upon examining the initial set of prompts, it is evident that they share a common characteristic of attempting to guide the model towards generating specific unintended outputs. Accordingly, they were correctly classified as injections.
# 2. Furthermore, there are only four instances assumed to be inaccurately predicted as non-injections. 
# 3. When evaluating these prompts, we are inclined to believe that they do not truly reflect real injection instances, suggesting a potential presence of mislabeled samples within the dataset.
# 4. Consequently, we think our model performed satisfactorily on the testing samples. However, a more in-depth investigation is necessary to explain why these prompts were initially categorized as prompt injections.
