# Component 10 — Test Generator

## Identity

```
Type:                   Transformer — Decoder (Autoregressive)
Parameters:             ~71M
Role:                   Generates test cases to verify produced code
                        Closes the verification loop — generate, test, confirm
                        Takes generated code + original specification as input
                        Outputs executable test code
```

---

## Hyperparameters

```
d_model:                512
n_heads:                8
d_head:                 64
n_layers:               12
d_ff:                   2048
dropout:                0.1
max_code_len:           512        generated code input
max_spec_len:           256        original specification input
max_combined_len:       768        code + spec concatenated
max_test_len:           512        generated test output
vocab_size:             32,000
activation:             GELU
norm:                   LayerNorm
bias:                   False

test_types:             4
  0  unit_test          single function input/output
  1  edge_case_test     boundary values and empty inputs
  2  performance_test   timing at multiple input sizes
  3  integration_test   multi-function interaction
```

---

## Parameter Count Verification

```
Embedding layer
  token embedding          32,000 × 512               =  16.384M
  position embedding       768 × 512                  =   0.393M
    (max_combined_len for input)
  test position embedding  512 × 512                  =   0.262M
    (max_test_len for output)
  embedding total                                     =  17.039M

Per transformer layer
  Attention
    Q projection           512 × 512                  =   0.262M
    K projection           512 × 512                  =   0.262M
    V projection           512 × 512                  =   0.262M
    output projection      512 × 512                  =   0.262M
    attention total                                   =   1.048M
  FFN
    up projection          512 × 2048                 =   1.049M
    down projection        2048 × 512                 =   1.049M
    ffn total                                         =   2.098M
  Layer norms (×2)                                    =   0.002M
  Per layer total                                     ≈   3.148M

All 12 layers              12 × 3.148M                =  37.776M

Test type head             512 × 4                    =   0.002M
Output lm head             512 × 32,000               =  16.384M

Grand Total                                           ≈  71.201M ✓
```

---

## Dimensional Walkthrough

```
INPUT A — Generated Code
  code token ids           (8, 512)
    from Component 08 output
    B = 8, code_len = 512

INPUT B — Specification
  spec token ids           (8, 256)
    original natural language request
    spec_len = 256

EMBEDDING
  code embeddings          (8, 512, 512)
  spec embeddings          (8, 256, 512)

SEPARATOR TOKEN
  insert SEP token between code and spec
  code + SEP + spec        (8, 769, 512)
    512 code + 1 SEP + 256 spec
  truncate to combined max (8, 768, 512)

POSITION EMBEDDINGS
  combined positions       (1, 768, 512)
  add to combined          (8, 768, 512)

ENCODER PASS (layers 1-6)
  full bidirectional attention on combined input
  per layer attention      (8, 8, 768, 768)
    B, n_heads, combined_len, combined_len
  per layer output         (8, 768, 512)
  encoder output           (8, 768, 512)
    encodes full understanding of code + spec

ENCODER POOLED
  mean pool                (8, 512)
    → conditioning signal for decoder

TEST TYPE HEAD
  linear                   (8, 512) → (8, 4)
  softmax                  (8, 4)
    → determines what type of test to generate first

DECODER PASS (layers 7-12)
  autoregressive test generation
  conditioned on encoder output via cross attention

  test token embedding     (8, T_test, 512)
    T_test grows from 0 to max_test_len = 512

  per decoder layer:
    masked self attention  (8, T_test, 512)
      causal mask on test tokens
    cross attention
      query from test      (8, T_test, 512)
      key from encoder     (8, 768, 512)
      value from encoder   (8, 768, 512)
      weights              (8, 8, T_test, 768)
        each test token attends to all code+spec tokens
      output               (8, T_test, 512)
    FFN                    (8, T_test, 512)

AFTER ALL 12 LAYERS
  hidden states            (8, T_test, 512)

FINAL LAYER NORM
  output                   (8, T_test, 512)

OUTPUT HEAD
  lm head                  (8, T_test, 512) → (8, T_test, 32000)
  logits at last position  (8, 1, 32000)
  sample / argmax          (8, 1)
  repeat until EOS         T_test → max 512

FINAL TEST OUTPUT
  generated test ids       (8, 512)
    executable Python test code
```

---

## Verification Loop

```
Generated test code        (8, 512) token ids
Decode to string           string   Python test code
Execute via subprocess     stdout, stderr, return_code
Parse results              pass_count, fail_count, error_count

If all tests pass:
  → generation complete
  → pass results to PRM as positive signal

If tests fail:
  → pass failure details back to Component 08
  → Generation Head conditions on failure signal
  → regenerate code with correction
  → repeat up to max_retries = 3
```

---

## Training Objective

```
Primary loss:             Cross entropy on next test token prediction
Secondary loss:           Cross entropy on test type classification
Ground truth:             (code, specification, test_suite) triples
                          from GitHub repos with existing test files
                          from competitive programming with known test cases
```

---

## Outputs Fed To

```
→  Component 06   Process Reward Model
     test results (pass/fail/error) as verification signal
     strongest ground truth signal in entire suite

→  Component 08   Code Generation Head
     failure details if tests fail
     triggers code revision loop
```

---

## Notes

```
- Encoder-decoder architecture unlike other components
  which are encoder-only or decoder-only
  justified: input (code+spec) and output (tests) are
  clearly separate modalities with different lengths
- Layers 1-6 act as encoder (bidirectional)
  Layers 7-12 act as decoder (causal + cross attention)
  single model with role determined by attention masking
- Test execution runs in isolated subprocess
  timeout = 5 seconds per test
  memory limit = 512MB per test
- Performance tests run at input sizes [10, 100, 1000]
  timing recorded and compared to complexity prediction
  from Component 05 — cross-component consistency check
```