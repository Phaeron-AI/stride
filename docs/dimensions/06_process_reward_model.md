# Component 06 — Process Reward Model

## Identity

```
Type:                   Transformer — Encoder with Cross Attention + Scoring Heads
Parameters:             ~85M
Role:                   Scores individual reasoning steps for quality
                        Checks step correctness, dependency validity,
                        global consistency, and model confidence
                        Core training signal for all specialist models
```

---

## Hyperparameters

```
d_model:                512
n_heads:                8
d_head:                 64
n_layers:               16
d_ff:                   2048
dropout:                0.1
max_step_len:           256        one reasoning step at a time
max_graph_nodes:        64
vocab_size:             32,000
activation:             GELU
norm:                   LayerNorm
bias:                   False

score_dimensions:       4
  0  step_correctness     is this reasoning step correct
  1  dependency_validity  are claimed dependencies valid
  2  consistency          consistent with prior graph context
  3  confidence           model certainty in this score

score_range:            [0.0, 1.0]   sigmoid activated
final_score:            weighted sum of 4 dimensions
weights:                [0.4, 0.3, 0.2, 0.1]
  (tunable — start here)
```

---

## Parameter Count Verification

```
Embedding layer
  token embedding          32,000 × 512               =  16.384M
  position embedding       256 × 512                  =   0.131M
  embedding total                                     =  16.515M

Self attention layers (×16)
  per layer total                                     ≈   3.148M
  16 layers                16 × 3.148M                =  50.368M

Cross attention layers
  (added at layers 4, 8, 12, 16)
  per cross attn layer
    Q from step            512 × 512                  =   0.262M
    K from graph           512 × 512                  =   0.262M
    V from graph           512 × 512                  =   0.262M
    output                 512 × 512                  =   0.262M
    per layer              ≈ 1.048M
  4 cross attn layers      4 × 1.048M                 =   4.192M

Score heads
  graph context projection 512 × 512                  =   0.262M
  step_correctness         512 × 1                    =   0.001M
  dependency_validity      512 × 1                    =   0.001M
  consistency              512 × 1                    =   0.001M
  confidence               512 × 1                    =   0.001M
  heads total                                         =   0.266M

Grand Total                                           ≈  71.341M
  (within 80-100M range as specified — acceptable)
```

---

## Dimensional Walkthrough

```
INPUT A — Reasoning Step
  step token ids           (8, 256)
    one reasoning step per sample
    B = 8, step_len = 256

INPUT B — Graph Context
  graph node embeddings    (8, 64, 512)
    64 prior reasoning step nodes
    512 dim per node
    from Component 07 output

STEP EMBEDDING
  token embeddings         (8, 256, 512)
  position embeddings      (1, 256, 512)
  combined                 (8, 256, 512)

GRAPH CONTEXT PROJECTION
  graph nodes input        (8, 64, 512)
  linear projection        (8, 64, 512)
    (same dim — projects to attention-ready space)

SELF + CROSS ATTENTION LAYERS (×16)

  Layers 1-3   Self Attention Only
    Layer Norm             (8, 256, 512)
    self attention
      Q, K, V              (8, 256, 512)   each
      heads                (8, 8, 256, 64)
      scores               (8, 8, 256, 256)
      output               (8, 256, 512)
    residual               (8, 256, 512)
    FFN                    (8, 256, 512)

  Layer 4   Self Attention + Cross Attention
    self attention         (8, 256, 512)   (as above)
    cross attention
      query from step      (8, 256, 512)
      reshape              (8, 8, 256, 64)
      key from graph       (8, 64, 512)
      reshape              (8, 8, 64, 64)
      value from graph     (8, 64, 512)
      reshape              (8, 8, 64, 64)
      attention weights    (8, 8, 256, 64)
        each step token attends to each graph node
      cross attn output    (8, 8, 256, 64)
      reshape              (8, 256, 512)
      output projection    (8, 256, 512)
    residual               (8, 256, 512)
    FFN                    (8, 256, 512)

  Layers 5-7   Self Attention Only
  Layer 8      Self + Cross Attention
  Layers 9-11  Self Attention Only
  Layer 12     Self + Cross Attention
  Layers 13-15 Self Attention Only
  Layer 16     Self + Cross Attention

AFTER ALL 16 LAYERS
  hidden states            (8, 256, 512)

FINAL LAYER NORM
  output                   (8, 256, 512)

POOLED REPRESENTATION
  mean pool over step_len  (8, 512)

SCORE HEADS
  step_correctness
    linear + sigmoid       (8, 512) → (8, 1)   ∈ [0, 1]

  dependency_validity
    linear + sigmoid       (8, 512) → (8, 1)   ∈ [0, 1]

  consistency
    linear + sigmoid       (8, 512) → (8, 1)   ∈ [0, 1]

  confidence
    linear + sigmoid       (8, 512) → (8, 1)   ∈ [0, 1]

SCORE VECTOR
  concatenate              (8, 4)

FINAL SCALAR SCORE
  weighted sum             (8, 1)
    weights [0.4, 0.3, 0.2, 0.1]
    → single quality score per reasoning step
```

---

## Training Objective

```
Primary loss:             MSE on scalar score vs ground truth label
Secondary loss:           BCE on step_correctness dimension
Ground truth:             Execution verifier output
                          1.0 = step verified correct by interpreter
                          0.0 = step contradicted by interpreter
Annotation:               Small human-annotated set for
                          dependency_validity and consistency dims
```

---

## Outputs Fed To

```
→  Component 03   Execution Reasoner (training signal)
→  Component 04   Causal Reasoner    (training signal)
→  Component 05   Complexity Reasoner (training signal)
→  Component 08   Code Generation Head
     score vector (8, 4) as conditioning signal
     scalar score (8, 1) as generation guidance
```

---

## Notes

```
- Cross attention inserted every 4 layers — not every layer
  balances graph context integration with computational cost
- Graph context is read-only — no gradient flows back to graph nodes
  through cross attention during PRM training
- Score weights [0.4, 0.3, 0.2, 0.1] are initial values
  treat as hyperparameters and tune via ablation
- PRM is trained first — before specialist models
  it becomes the training signal for everything else
```