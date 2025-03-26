"This script:
 1. Loads training and test dataset from `data` folder;
 2. Embedding text with BERT;
 3. Train embedddings with several ML models;
 4. Save models in `models` folder.
 5. Print model metrics.
 
See section 'Usage' in READMD.md to setup the environment."

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

(defn gen-sentence-embedding [tokenizer bert-model prompt]
  (setv tokens (tokenizer prompt
                          :padding True
                          :truncation True
                          :return-tensors "pt"
                          :add-special-tokens True))
  (with [_ (torch.no-grad)]
    (setv output (bert-model #** tokens)))
  (-> output.last-hidden-state
    (.mean :dim 1)
    .squeeze
    .numpy))

(defn load-data [root-folder]
  (let [ds-train (pl.read_parquet
                   f"{root-folder}/deepset/data/train-00000-of-00001-9564e8b05b4757ab.parquet")
        ds-test (pl.read_parquet
                  f"{root-folder}/deepset/data/test-00000-of-00001-701d16158af87368.parquet")
        sf-train (pl.read_parquet f"{root-folder}/safe-guard/data/train-00000-of-00001.parquet")
        sf-test (pl.read_parquet f"{root-folder}/safe-guard/data/test-00000-of-00001.parquet")
        btoz (BertTokenizer.from_pretrained "bert-base-multilingual-uncased")
        bmod (BertModel.from_pretrained "bert-base-multilingual-uncased")]
    {:x-train 
     (-> (pl.concat [ds-train sf-train])
       (get "text")
       (.map-elements (fn [p] (gen-sentence-embedding btoz bmod p)))
       (.to-list)
       (pl.from-records :orient "row"))
     :x-test
     (-> (pl.concat [ds-test sf-test])
       (get "text")
       (.map-elements (fn [p] (gen-sentence-embedding btoz bmod p)))
       (.to-list)
       (pl.from-records :orient "row"))
     :y-train (get (pl.concat [ds-train sf-train]) "label")
     :y-test (get (pl.concat [ds-test sf-test]) "label")}))
  
(defn eval-model [dataset estimator model model-path]
  (let [y-test (get dataset :y-test)
        y-pred (do (.fit model (get dataset :x-train) (get dataset :y-train))
                   (joblib.dump model model-path)
                   (.predict model (get dataset :x-test)))]
    {"Estimator" estimator
     "Accuracy" (accuracy_score y-test y-pred)
     "Precision" (precision_score y-test y-pred)
     "Recall" (recall_score y-test y-pred)
     "F1 Score" (f1_score y-test y-pred)}))

(let [model-base-path (Path "./models")
      data (load-data "./data")    
      estimators {:naive-bayes (GaussianNB)
                  :logistic-regression (LogisticRegression)
                  :support-vector-machine (SVC)
                  :random-forest (RandomForestClassifier)}
      result
      (pl.from-dicts
        (lfor
          ename estimators
          :setv emod (get estimators ename)
          (eval-model data
                      ename
                      emod
                      (.joinpath
                       model-base-path
                       f"{ename}.pkl"))))]
  
  (print f"Analysis result:\n{result}"))
