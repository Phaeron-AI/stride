# DERIVE — Component Interfaces

> The critical joins between components.
> Get these wrong and nothing connects.
> Verify these shapes before writing any integration code.

---

## Interface Map

```
01 Execution Tracer
   └─→ 02 Tokenizer
          └─→ 03 Execution Reasoner
          └─→ 04 Causal Reasoner

03 Execution Reasoner  ─→ 07 Reasoning Graph Builder
04 Causal Reasoner     ─→ 07 Reasoning Graph Builder
05 Complexity Reasoner ─→ 07 Reasoning Graph Builder

07 Reasoning Graph Builder ─→ 06 Process Reward Model
07 Reasoning Graph Builder ─→ 08 Code Generation Head

06 Process Reward Model ─→ 08 Code Generation Head (conditioning)
06 Process Reward Model ─→ 03, 04, 05 (training signal)

08 Code Generation Head ─→ 09 Syntax Verifier (rolling window)
08 Code Generation Head ─→ 10 Test Generator (complete output)

09 Syntax Verifier ─→ 08 Code Generation Head (error feedback)
10 Test Generator  ─→ 08 Code Generation Head (test feedback)
10 Test Generator  ─→ 06 Process Reward Model (verification signal)
```

---

## Interface 01 → 02

```
FROM    Execution Tracer
TO      Tokenizer

Signal  raw execution trace
Shape   (T, V)
  T = execution steps, variable, max 512
  V = variables tracked, variable, max 64
Type    List[Dict] before serialization
```

---

## Interface 02 → 03

```
FROM    Tokenizer
TO      Execution Reasoner

Signal  token embeddings
Shape   (8, 512, 512)
  B=8, seq_len=512, token_dim=512
Type    torch.Tensor, BF16
```

---

## Interface 02 → 04

```
FROM    Tokenizer
TO      Causal Reasoner

Signal A  token embeddings
Shape     (8, 512, 512)
  same as above

Signal B  AST dependency features
Shape     (8, 512, 64)
  B=8, seq_len=512, ast_feature_dim=64
Type      torch.Tensor, BF16
Note      AST features extracted separately via ast module
          not produced by tokenizer — joined here
```

---

## Interface 03 → 07

```
FROM    Execution Reasoner
TO      Reasoning Graph Builder

Signal  pooled representation
Shape   (8, 768)   before projection
        (8, 512)   after projection inside Graph Builder
Type    torch.Tensor, BF16
Note    mean pool over seq_len dimension of final hidden states
```

---

## Interface 04 → 07

```
FROM    Causal Reasoner
TO      Reasoning Graph Builder

Signal A  pooled representation
Shape     (8, 768) → projected to (8, 512) inside Graph Builder

Signal B  node embeddings
Shape     (8, 512, 768)
  B=8, seq_len=512, d_model=768
  one embedding per token position

Signal C  edge predictions (adjacency)
Shape     (8, 512, 512)
  B=8, seq_len×seq_len pairwise scores
  values in [0, 1] after sigmoid

Signal D  edge type predictions
Shape     (8, 512, 512, 6)
  B=8, seq_len×seq_len, edge_types=6
  values are softmax probabilities
Type      torch.Tensor, BF16
```

---

## Interface 05 → 07

```
FROM    Complexity Reasoner
TO      Reasoning Graph Builder

Signal A  pooled representation
Shape     (8, 512)
  no projection needed — already d_model_small

Signal B  time complexity
Shape     (8, 8)
  probability distribution over 8 classes

Signal C  space complexity
Shape     (8, 8)
  probability distribution over 8 classes

Signal D  bottleneck locations
Shape     (8, 512)
  soft attention over token positions
Type      torch.Tensor, BF16
```

---

## Interface 07 → 06

```
FROM    Reasoning Graph Builder
TO      Process Reward Model

Signal A  node embeddings
Shape     (8, 64, 512)
  B=8, max_nodes=64, d_model_small=512

Signal B  adjacency matrix
Shape     (8, 64, 64)
  binary — 1 = edge exists, 0 = no edge

Signal C  node types
Shape     (8, 64, 4)
  softmax probabilities over 4 node types
Type      torch.Tensor, BF16
```

---

## Interface 07 → 08

```
FROM    Reasoning Graph Builder
TO      Code Generation Head

Signal  node embeddings
Shape   (8, 64, 512)   from Graph Builder output
        (8, 64, 768)   after projection inside Generation Head
  projected to match d_model_large for cross attention
Type    torch.Tensor, BF16
```

---

## Interface 06 → 08

```
FROM    Process Reward Model
TO      Code Generation Head

Signal A  score vector
Shape     (8, 4)
  four dimension scores, each in [0, 1]

Signal B  scalar score
Shape     (8, 1)
  weighted sum of score vector
  used as generation guidance strength
Type      torch.Tensor, BF16
Note      injected at embedding level in Generation Head
          projected: (8, 4) → (8, 768) via linear layer
```

---

## Interface 08 → 09

```
FROM    Code Generation Head
TO      Syntax Verifier

Signal  partial generated tokens
Shape   (8, 256)
  rolling window — last 256 generated tokens
  if T < 256 then pad to (8, 256)
  updated every 16 generation steps
Type    torch.LongTensor (token ids, not embeddings)
```

---

## Interface 09 → 08

```
FROM    Syntax Verifier
TO      Code Generation Head

Signal A  syntax valid
Shape     (8, 1)
  scalar in [0, 1]
  threshold: < 0.5 triggers correction

Signal B  error location
Shape     (8, 256)
  soft attention over window positions
  (only used when syntax_valid < 0.5)

Signal C  error type
Shape     (8, 12)
  probability over 12 error classes
  (only used when syntax_valid < 0.5)
Type      torch.Tensor, BF16
```

---

## Interface 08 → 10

```
FROM    Code Generation Head
TO      Test Generator

Signal  complete generated tokens
Shape   (8, 1024)   full generation output
        (8, 512)    truncated to max_code_len for Test Generator input
Type    torch.LongTensor (token ids)
Note    truncate from the end if generation > 512
        code structure front-loaded — truncating end loses less
```

---

## Interface 10 → 08

```
FROM    Test Generator
TO      Code Generation Head

Signal  test execution results
Type    Dict {
          pass_count:   int
          fail_count:   int
          error_count:  int
          failed_tests: List[str]   failed test names
          error_msgs:   List[str]   error messages
        }
Note    not a tensor — structured Python dict
        triggers code revision loop if fail_count > 0
        max_retries = 3 before declaring failure
```

---

## Interface 10 → 06

```
FROM    Test Generator
TO      Process Reward Model

Signal  verification outcome
Type    float in [0.0, 1.0]
  pass_count / total_tests
  1.0 = all tests pass
  0.0 = all tests fail
Note    strongest ground truth signal in suite
        used to update PRM weights during online learning
```

---

## Interface 06 → 03, 04, 05

```
FROM    Process Reward Model
TO      Execution Reasoner, Causal Reasoner, Complexity Reasoner

Signal  step quality scores
Shape   (8, 1)   scalar per sample per reasoning step
Type    float in [0.0, 1.0]
Note    training signal only — not used during inference
        specialist models trained to produce reasoning steps
        that receive high PRM scores
```

---

## Quick Reference — All Tensor Shapes

```
Component                Output Shape              Notes
──────────────────────────────────────────────────────────────────
01 Execution Tracer      (T, V)                    variable T, V
02 Tokenizer             (8, 512, 512)             BF16
03 Execution Reasoner    (8, 768) pooled           BF16
04 Causal Reasoner       (8, 768) pooled           BF16
                         (8, 512, 512) adj         BF16
                         (8, 512, 512, 6) types    BF16
05 Complexity Reasoner   (8, 512) pooled           BF16
                         (8, 8) time complexity    BF16
                         (8, 8) space complexity   BF16
                         (8, 512) bottleneck       BF16
06 PRM                   (8, 4) score vector       BF16
                         (8, 1) scalar score       BF16
07 Graph Builder         (8, 64, 512) nodes        BF16
                         (8, 64, 64) adjacency     BF16
                         (8, 64, 4) node types     BF16
08 Generation Head       (8, 1024) token ids       LongTensor
09 Syntax Verifier       (8, 1) valid              BF16
                         (8, 256) error location   BF16
                         (8, 12) error type        BF16
10 Test Generator        (8, 512) test token ids   LongTensor
                         Dict results              Python dict
──────────────────────────────────────────────────────────────────
```
