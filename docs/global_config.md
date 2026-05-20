# STRIVE — Global Configuration

## System Identity

```
Model Name:             STRIVE
Full Name:              Structured Reasoning with Integrated Depth                   Engine
Version:                0.1.0
Hardware:               NVIDIA RTX 5070 Ti Laptop
VRAM:                   12.8GB
Architecture:           Blackwell
```

---

## Global Hyperparameters

```
Precision:              BF16
Vocabulary Size:        32,000
Max Sequence Length:    2,048
Batch Size:             8
Gradient Accumulation:  8 steps
Effective Batch Size:   64
Optimizer:              AdamW
Learning Rate:          3e-4
Warmup Steps:           1,000
Device:                 CUDA
```

---

## Shared Dimensions

```
d_model_large:          768        (used by large specialist models)
d_model_small:          512        (used by small specialist models)
d_model_tiny:           256        (used by verifier)
d_ff_large:             3072       (4 × 768)
d_ff_small:             2048       (4 × 512)
d_ff_tiny:              1024       (4 × 256)
d_head:                 64         (consistent across all components)
max_seq_len:            512        (specialist models)
max_gen_len:            1024       (generation head)
max_step_len:           256        (PRM)
max_graph_nodes:        64
```

---

## Shared Vocabulary

```
vocab_size:             32,000
tokenizer:              code-focused BPE
special tokens:
  PAD:                  0
  BOS:                  1
  EOS:                  2
  SEP:                  3          (separates code and context)
  STEP:                 4          (marks reasoning step boundary)
  NODE:                 5          (marks graph node boundary)
```

---

## Edge Types (Shared Across Components)

```
0   supports
1   contradicts
2   refines
3   branches
4   depends_on
5   produces
```

## Node Types (Shared Across Components)

```
0   observation
1   inference
2   hypothesis
3   conclusion
```

---

## VRAM Budget

```
Component                  Params     VRAM BF16
────────────────────────────────────────────────
Execution Reasoner         219M       438MB
Causal Reasoner            219M       438MB
Complexity Reasoner        83M        166MB
Process Reward Model       85M        170MB
Reasoning Graph Builder    44M         88MB
Code Generation Head       219M       438MB
Syntax Verifier            22M         44MB
Test Generator             71M        142MB
────────────────────────────────────────────────
Specialist Total           962M       ~1.9GB
Language Backbone (4-bit)  14B        ~8.0GB
────────────────────────────────────────────────
Peak Inference Total                  ~9.9GB 
────────────────────────────────────────────────
```