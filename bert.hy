(require hyrule [->])
(import torch
        polars :as pl
        sklearn.ensemble [RandomForestClassifier]
        sklearn.linear_model [LogisticRegression]
        sklearn.metrics [accuracy_score precision_score recall_score f1_score]
        sklearn.naive_bayes [GaussianNB]
        sklearn.svm [SVC]
        transformers [BertTokenizer BertModel])

(defn gen-sentence-embedding [prompt]
  (setv tokens (tokenizer prompt
                         :padding True
                         :truncation True
                         :return-tensors "pt"
                         :add-special-tokens True))
  (with [_ (torch.no-grad)]
    (setv output (model #** tokens)))
  (-> output.last-hidden-state
    (.mean :dim 1)
    .squeeze
    .numpy))

(defn eval-model [estimator]
  (let [model (get estimators estimator)
        y-pred (do (.fit model x-train (get train "label"))
                   (.predict model x-test))]
    {"Estimator" estimator
     "Accuracy" (accuracy_score (get test "label") y-pred)
     "Precision" (precision_score (get test "label") y-pred)
     "Recall" (recall_score (get test "label") y-pred)
     "F1 Score" (f1_score (get test "label") y-pred)}))

(setv
  tokenizer (BertTokenizer.from_pretrained "bert-base-multilingual-uncased")
  model (BertModel.from_pretrained "bert-base-multilingual-uncased")

  train (.rename
          (pl.read_parquet
            "./data/data/train-00000-of-00001-9564e8b05b4757ab.parquet")
          {"text" "prompt"})

  test (.rename
         (pl.read_parquet
           "./data/data/test-00000-of-00001-701d16158af87368.parquet")
         {"text" "prompt"})

  x-train
  (-> train
    (get "prompt")
    (.map-elements gen-sentence-embedding)
    (.to-list)
    (pl.from-records :orient "row"))

  x-test
  (-> test
    (get "prompt")
    (.map-elements gen-sentence-embedding)
    (.to-list)
    (pl.from-records :orient "row"))

  estimators {"Naive Bayes" (GaussianNB)
              "Logistic Regression" (LogisticRegression)
              "Support Vector Machine" (SVC)
              "Random Forest" (RandomForestClassifier)}

  result (pl.from-dicts
           (lfor est estimators (eval-model est))))

(print f"Analysis result:\n{result}")
