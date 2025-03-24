"This script:
 1. Loads training and test dataset from `data` folder;
 2. Embedding text with BERT;
 3. Train embedddings with several ML models;
 4. Save models in `models` folder.
 5. Print model metrics."

(require hyrule [->])
(import joblib
        pathlib [Path]
        polars :as pl
        sklearn.ensemble [RandomForestClassifier]
        sklearn.linear_model [LogisticRegression]
        sklearn.metrics [accuracy_score precision_score recall_score f1_score]
        sklearn.naive_bayes [GaussianNB]
        sklearn.svm [SVC]
        torch
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

(defn eval-model [estimator model-path]
  (let [model (get estimators estimator)
        y-pred (do (.fit model x-train (get train "label"))
                   (joblib.dump model model-path)
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

  estimators {"naive-bayes" (GaussianNB)
              "logistic-regression" (LogisticRegression)
              "support-vector-machine" (SVC)
              "random-forest" (RandomForestClassifier)}

  result (pl.from-dicts
           (let [model-base-path (Path "./models")]
             (.mkdir model-base-path :exist-ok True)
             (lfor est estimators
               (eval-model est (.joinpath
                                model-base-path
                                f"{est}.pkl"))))))

(print f"Analysis result:\n{result}")
