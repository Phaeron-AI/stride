# Component 08 — Code Generation Head

## Identity

```
Type:                   Transformer — Decoder (Autoregressive)
Parameters:             ~219M
Role:                   Generates code conditioned on reasoning graph
                        Does not generate blindly — follows graph structure
                        Cross attends to graph nodes every 4 layers
```

---

## Hyperparameters

```
d_model:                768
n_heads:                12
d_head:                 64
n_layers:               24
d_ff:                   3072
dropout:                0.1
max_prompt_len:         256
max_gen_len:            1024
vocab_size:             32,000
activation:             GELU
norm:                   LayerNorm
bias:                   False

graph_node_dim:         512        input dim from Graph Builder
graph_proj_dim:         768        projected to match d_model
graph_cross_attn_every: 4          cross attend to graph every N layers
```

---

## Parameter Count Verification

```
Embedding layer
  token embedding          32,000 × 768               =  24.576M
  position embedding       1024 × 768                 =   0.786M
  embedding total                                     =  25.362M

Self attention layers (×24)
  per layer total                                     ≈   7.081M
  24 layers                24 × 7.081M                = 169.944M

Graph cross attention layers
  (at layers 4, 8, 12, 16, 20, 24 — 6 total)
  per cross attn layer
    Q from tokens          768 × 768                  =   0.590M
    K from graph           768 × 768                  =   0.590M
    V from graph           768 × 768                  =   0.590M
    output projection      768 × 768                  =   0.590M
    per layer              ≈ 2.360M
  6 cross attn layers      6 × 2.360M                 =  14.160M

Graph projection
  node dim to d_model      512 × 768                  =   0.393M

PRM conditioning
  score projection         4 × 768                    =   0.003M

Output head
  lm head                  768 × 32,000               =  24.576M

Grand Total                                           ≈ 234.438M
  (reported as ~219M base + ~15M cross attention heads)
```

---

## Dimensional Walkthrough

```
INPUT A — Prompt
  prompt token ids         (8, 256)
    natural language specification
    B = 8, prompt_len = 256

INPUT B — Graph Conditioning
  node embeddings          (8, 64, 512)    from Component 07

INPUT C — PRM Signal
  score vector             (8, 4)          from Component 06

GRAPH PROJECTION
  linear project           (8, 64, 512) → (8, 64, 768)
    match d_model for cross attention

PRM SIGNAL INJECTION
  linear project           (8, 4) → (8, 768)
  expand                   (8, 1, 768)
  (will be added to prompt embedding)

PROMPT EMBEDDING
  token embeddings         (8, 256, 768)
  position embeddings      (1, 256, 768)
  PRM signal               (8, 1, 768)    broadcast add to all positions
  combined                 (8, 256, 768)

CAUSAL MASK
  lower triangular mask    (256, 256)     prevents attending to future tokens
  extended during generation to (T, T) as T grows

AUTOREGRESSIVE GENERATION LOOP
  sequence length T grows from 256 to max 1024

  LAYERS 1-3   Masked Self Attention Only
    Layer Norm             (8, T, 768)
    masked self attention
      Q, K, V              (8, T, 768)    each
      heads                (8, 12, T, 64)
      causal scores        (8, 12, T, T)
      masked softmax       (8, 12, T, T)  future tokens masked to -inf
      output               (8, T, 768)
    residual               (8, T, 768)
    FFN                    (8, T, 768)

  LAYER 4   Masked Self Attention + Graph Cross Attention
    self attention         (8, T, 768)    (as above)

    graph cross attention
      query from tokens    (8, T, 768)
      heads                (8, 12, T, 64)
      key from graph       (8, 64, 768)
      heads                (8, 12, 64, 64)
      value from graph     (8, 64, 768)
      heads                (8, 12, 64, 64)
      attention weights    (8, 12, T, 64)
        each token position attends to each of 64 graph nodes
        no masking — graph nodes are fully visible
      cross attn output    (8, 12, T, 64)
      reshape              (8, T, 768)
      output projection    (8, T, 768)
    residual               (8, T, 768)
    FFN                    (8, T, 768)

  LAYERS 5-7   Self Attention Only
  LAYER 8      Self + Graph Cross Attention
  LAYERS 9-11  Self Attention Only
  LAYER 12     Self + Graph Cross Attention
  LAYERS 13-15 Self Attention Only
  LAYER 16     Self + Graph Cross Attention
  LAYERS 17-19 Self Attention Only
  LAYER 20     Self + Graph Cross Attention
  LAYERS 21-23 Self Attention Only
  LAYER 24     Self + Graph Cross Attention

AFTER ALL 24 LAYERS
  hidden states            (8, T, 768)
    T = current generation length

FINAL LAYER NORM
  output                   (8, T, 768)

OUTPUT HEAD
  lm head linear           (8, T, 768) → (8, T, 32000)
  logits at last position  (8, 1, 32000)
    only last position needed for next token prediction

NEXT TOKEN
  sample or argmax         (8, 1)
  append to sequence       T → T + 1
  repeat until EOS or max_gen_len

FINAL OUTPUT
  generated token ids      (8, 1024)   padded to max_gen_len
```

---

## Training Objective

```
Primary loss:             Cross entropy on next token prediction
                          teacher forcing during training
Conditioning:             Graph nodes always provided
                          PRM score from training step PRM evaluation
Ground truth:             (specification, reasoning_graph, correct_code) triples
                          constructed from GitHub + synthetic generation
```

---

## Outputs Fed To

```
→  Component 09   Syntax Verifier
     partial generated tokens  (8, 256)   rolling window

→  Component 10   Test Generator
     complete generated tokens (8, 1024)  full generated code
```

---

## Notes

```
- Causal (lower triangular) mask applied to self attention only
  graph cross attention has no mask — nodes fully visible always
- Cross attention every 4 layers balances
  graph guidance strength vs computational cost
- PRM score injected at embedding level — simple but effective
  alternative: use PRM as beam search reranker (future work)
- KV cache used during inference to avoid recomputing
  past key/value pairs at each generation step
  critical for generation speed on single GPU
- T grows from prompt_len (256) to max_gen_len (1024)
  VRAM for attention scales as O(T²) — monitor during long generation
```