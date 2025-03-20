(setv tokens
  (tokenizer "How to study Prompt Injection of LLM?"  ; see API doc with `?tokenizer.__call__`
    :padding True
    :truncation True
    :return-tensors "pt"
    :add-special-tokens True))

(.keys demo-enc) ; (dict-keys ["input_ids" "token_type_ids" "attention_mask"])
(. (:input-ids demo-enc) shape) ; torch.Size([1, 13])
(:attention-mask demo-enc) ; tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

(with [_ (torch.no-grad)]
  (setv output (model #** tokens)))
(. output.last-hidden-state shape) ; torch.Size([1, 13, 768]), each token is converted to 768-vector
(let [sentence-embedding
      (-> output.last-hidden-state
        (.mean :dim 1)
        .squeeze
        .numpy)]
  (. sentence-embedding shape)) ; (768,), the embedding of a sentence is the mean of embeddings of each token


(.count train)
(.len Xtrain)
(. (get (.head Xtrain 1) 0) shape)

(setv demo-emb (gen-sentence-embedding "I almost finish it"))
(. demo-emb dtype)
