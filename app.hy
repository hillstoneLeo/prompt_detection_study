(import torch
        random
        polars :as pl
        transformers [BertTokenizer BertModel AutoTokenizer]
        sklearn.metrics.pairwise [cosine_similarity])
(require hyrule [-> ->> as-> some-> doto])

(setv train (.rename
              (pl.read_parquet
                "./data/data/train-00000-of-00001-9564e8b05b4757ab.parquet")
              {"text" "prompt"})
      test (.rename
             (pl.read_parquet
               "./data/data/test-00000-of-00001-701d16158af87368.parquet")
             {"text" "prompt"}))

(setv tokenizer (BertTokenizer.from_pretrained "bert-base-multilingual-uncased"))
(setv model (BertModel.from_pretrained "bert-base-multilingual-uncased"))

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

(setv Xtrain (.map-elements (get train "prompt") gen-sentence-embedding))
(setv Xtest (.map-elements (get test "prompt") gen-sentence-embedding))

