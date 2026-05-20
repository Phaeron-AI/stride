# Component 02 — Physical State Tokenizer

## Identity

```
Type:                   Deterministic Preprocessing Module
Parameters:             None (not a neural model)
Role:                   Converts raw execution traces into
                        token embeddings consumable by neural components
```

---

## Hyperparameters

```
vocab_size:             32,000
max_variables:          64
max_trace_length:       512
token_dim:              512        embedding dimension output
padding:                True       pad to max_trace_length
truncation:             True       truncate if exceeded
```

---

## Dimensional Walkthrough

```
INPUT
  raw trace                (T, V)
    T = execution steps    variable, max 512
    V = variables          variable, max 64

STEP 1 — FLATTEN
  flatten trace            (T × V,)
    1D sequence of variable values

STEP 2 — TOKENIZE
  token sequence           (T × V,)   each value → token id(s)

STEP 3 — CLIP / PAD
  clipped sequence         (512,)
    truncate if T×V > 512
    pad with PAD token if T×V < 512

STEP 4 — EMBED
  token embeddings         (512, 512)
    (seq_len, token_dim)

BATCHED OUTPUT
  token embeddings         (8, 512, 512)
    B   = batch size = 8
    512 = sequence length
    512 = token_dim (d_model_small)
```

---

## Outputs Fed To

```
→  Component 03   Execution Reasoner
     token embeddings (8, 512, 512)

→  Component 04   Causal Reasoner
     token embeddings (8, 512, 512)
     alongside AST features (8, 512, 64)
```

---

## Notes

```
- Variable values serialized to string before tokenization
- Numerical values normalized before serialization
- Token dim 512 matches d_model_small used by specialist models
- Embedding weights are learned during specialist model training
  not fixed in the tokenizer itself
```