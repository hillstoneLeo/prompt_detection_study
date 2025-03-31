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
        sklearn.metrics [accuracy_score
                         confusion_matrix
                         f1_score
                         precision_score
                         recall_score
                         roc_curve]
        sklearn.naive_bayes [GaussianNB]
        sklearn.svm [SVC]
        time [sleep]
        torch
        transformers [BertTokenizer BertModel]
        translate [Translator])

"Bert tokenizer & model need kept global for they're used in loops"
(setv btok (BertTokenizer.from_pretrained "bert-base-multilingual-uncased")
      bmod (BertModel.from_pretrained "bert-base-multilingual-uncased"))

;; Calculate FPR (误报率), FNR (漏报率) and other metrics 
;; based on `label` and `predict` values. Usage:
;; (setv res (pred-metrics [1 0 0 1 1, 0 0 1 0 1] [1 1 0 0 0, 1 0 0 1 0]))
(defn pred-metrics [label predict]
  (let [[[_ fpr _] [_ tpr _] thr]
        (roc_curve label predict)]
    {:fpr (.item fpr)
     :fnr (- 1 (.item tpr))
     :accurary (accuracy_score label predict)
     :precision (precision_score label predict)
     :recall (recall_score label predict)
     :f1-score (f1_score label predict)}))

;; Generate embeddings from `prompt` with `tokenizer`, `bert-model`
;; Usage:
;; (gen-sentence-embedding btok bmod p)
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

;; Load dataset deespset & safe-guard from `root-foler`.
;; Usage:
;; (load-data "./data")
(defn load-data [root-folder]
  (let [ds-train (pl.read_parquet
                   f"{root-folder}/deepset/data/train-00000-of-00001-9564e8b05b4757ab.parquet")
        ds-test (pl.read_parquet
                  f"{root-folder}/deepset/data/test-00000-of-00001-701d16158af87368.parquet")
        sf-train (pl.read_parquet f"{root-folder}/safe-guard/data/train-00000-of-00001.parquet")
        sf-test (pl.read_parquet f"{root-folder}/safe-guard/data/test-00000-of-00001.parquet")]
    {:x-train 
     (-> (pl.concat [ds-train sf-train])
       (get "text")
       (.map-elements (fn [p] (gen-sentence-embedding btok bmod p)))
       (.to-list)
       (pl.from-records :orient "row"))
     :x-test
     (-> (pl.concat [ds-test sf-test])
       (get "text")
       (.map-elements (fn [p] (gen-sentence-embedding btok bmod p)))
       (.to-list)
       (pl.from-records :orient "row"))
     :y-train (get (pl.concat [ds-train sf-train]) "label")
     :y-test (get (pl.concat [ds-test sf-test]) "label")}))

;; Usage:
;; (setv train-models "./data" "./models")
(defn train-models [data-path model-path]
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
  (let [model-base (Path model-path)
        data (load-data data-path)
        estimators {:naive-bayes (GaussianNB)
                    :logistic-regression (LogisticRegression)
                    :support-vector-machine (SVC)
                    :random-forest (RandomForestClassifier)}]
    (pl.from-dicts
      (lfor
        ename estimators
        :setv emod (get estimators ename)
        (eval-model data
                    ename
                    emod
                    (.joinpath
                     model-base
                     f"{(. ename name)}.pkl"))))))

;; Sample 100 English prompt from deepset & safe-guard dataset,
;; translate to another language (default: Chinese) with Google Translate.
;; Usage:
;; (setv res-series (translate-prompt "./data")))
(defn translate-prompt [data-path [outlang "zh"]]
  (defn trans-sentences [#^ pl.Series inp #^ Path cache-base]
    (setv trans-cache (/ cache-base (Path "trans-cache.pkl")))
    (if (.exists trans-cache)
        (pl.DataFrame {"text" (joblib.load trans-cache)})
        (do (setv res [])
            (for [s inp]
              (.append
                res
                (translator.translate s))
              (sleep 0.5))
            (joblib.dump res trans-cache)  
            (pl.DataFrame {"text" res}))))
  (let [translator (Translator :to-lang outlang)
        testdata 
        (-> (pl.concat
              [(pl.read_parquet
                 f"{data-path}/deepset/data/test-00000-of-00001-701d16158af87368.parquet")
               (pl.read_parquet
                 f"{data-path}/safe-guard/data/test-00000-of-00001.parquet")])
            (.sample :n 100 :seed 42)
            (.with-columns :lang (pl.lit "en")))
        translated
        (.with-columns
          (pl.concat :how "horizontal"
            [(trans-sentences (get testdata "text")
                  (Path data-path))
             (.select testdata "label")])
          :lang (pl.lit "zh"))]
    (pl.concat [testdata translated])))

;; Predict on a test dataset `data-path` with model `model-name`.
;; Usage:
;; (setv predict-metrics (predict-injection "./data/manual-all.tsv" "svm"))  
(defn predict-injection [#^ str data-path #^ str model-name]
  (let [model (joblib.load f"./models/{model-name}.pkl")
        test (pl.read-csv data-path :separator "\t")
        y-test (get test "label")
        y-pred (.predict
                 model
                 (-> test
                     (get "text")
                     (.map-elements (fn [p] (gen-sentence-embedding btok bmod p)))
                     (.to-list)
                     (pl.from-records :orient "row")))
        predict-res (.with-columns test :predict y-pred)
        [fpr _ _] (roc_curve y-test y-pred)]
    (.write-csv predict-res "injection-predict.tsv" :separator "\t")
    (pred-metrics y-test y-pred)))

