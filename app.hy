(import torch
        polars :as pl
        transformers [BertTokenizer BertModel])

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

  

