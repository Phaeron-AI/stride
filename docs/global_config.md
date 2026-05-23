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

## STRIDE — Project Summary

### What We're Building

A suite of small specialist neural models (~1B parameters total) that outperform monolithic 7B+ models on code reasoning and generation. Instead of one large model doing everything, STRIDE uses coordinated specialists that each solve a narrow problem extremely well.

**The core thesis:** vertical scaling through architecture beats horizontal scaling through parameters. A 1B specialist suite with the right inductive biases outperforms a 7B general model on code reasoning tasks.

**Five architectural innovations working together:**
```
Block AttnRes    bounded depth growth, better information flow
LoopLM           adaptive latent refinement, knowledge manipulation
Reasoning Graph  explicit structural representation of reasoning
Step-level PRM   verified correctness at every reasoning step
Execution Ground free ground truth from the Python interpreter
```

**Target hardware:** single RTX 5070 Ti (12.8GB VRAM)

---

### System Architecture

```
Language Backbone (DeepSeek-R1-Distill-14B, borrowed)
         ↓
Reasoning Graph Builder
         ↓
┌────────────────────────────────────┐
│  Execution    Causal    Complexity │
│  Reasoner     Reasoner  Reasoner  │
│  219M R=3     219M R=3  83M R=2   │
│  AttnRes N=8  AttnRes N=8 N=4     │
└────────────────────────────────────┘
         ↓
Process Reward Model (85M)
         ↓
Code Generation Head (219M, R=2)
         ↓
┌──────────────────────────────┐
│  Syntax Verifier  Test Gen   │
│  22M              71M        │
└──────────────────────────────┘
```

---

### What's Done

**Phase 0 — Config and fixes ✓**
```
engine/config/global_config.py       all component dataclasses
engine/config/components/*.yaml      12 component yaml files
```

**Phase 1 — Data pipeline ✓**
```
engine/data/execution_tracer.py      traces Python function execution
engine/data/tokenizer.py             converts traces to tensors
engine/data/pipeline.py              bulk dataset builder + DataLoader
```

**Phase 2.1 — Base transformer (in progress)**
```
engine/reasoning/components/
  rms_norm.py           ✓ complete
  rope.py               ✓ complete
  gqa.py                ✓ complete
  swiglu.py             ✓ complete
  block_attn_res.py     ✓ complete
  transformer_layer.py  ✓ complete
  exit_gate.py          ✓ complete
  __init__.py           ✓ complete

engine/reasoning/base_transformer.py  ⚠ 3 bugs to fix
  Bug 1: block boundary == 16 → == 0
  Bug 2: pseudo-query re-zeroing commented out
  Bug 3: loop_forward has no loop, never calls _one_loop_pass
```

---

### What's Left

**Phase 2 — Remaining specialists**
```
engine/reasoning/execution_reasoner.py    ExecutionReasoner(STRIDETransformer)
engine/reasoning/causal_reasoner.py       CausalReasoner(STRIDETransformer)
engine/reasoning/complexity_reasoner.py   ComplexityReasoner(STRIDETransformer)
engine/reasoning/graph_builder.py         ReasoningGraphBuilder
```

**Phase 3 — Reward model**
```
engine/reward/process_reward_model.py     ProcessRewardModel
  5 score dimensions including loop_improvement
  cross-attention to graph nodes every 4 layers
```

**Phase 4 — Generation**
```
engine/generation/code_generation_head.py  CodeGenerationHead(STRIDETransformer)
engine/generation/syntax_verifier.py       SyntaxVerifier
engine/generation/test_generator.py        TestGenerator
```

**Phase 5 — Training loops**
```
engine/training/trainer.py                Trainer base class
engine/training/prm_trainer.py            PRMTrainer
engine/training/specialist_trainer.py     SpecialistTrainer
engine/training/generation_trainer.py     GenerationTrainer
```

**Phase 6 — Evaluation**
```
engine/evaluation/benchmarks.py           BenchmarkRunner
  CRUXEval, CodeMind, custom suite
  comparison vs Qwen2.5-7B baseline
```

**Phase 7 — Integration**
```
engine/generation/language_backbone.py    Ollama wrapper for DeepSeek
main.py                                   full end-to-end pipeline
```

---

### Immediate Next Step

Fix the three bugs in `base_transformer.py` and run:

```bash
cd engine
python reasoning/base_transformer.py
```

Expected output:
```
Parameters: ~2.5M
✓ All pseudo-queries initialized to zero
loop 1 hidden: (2, 64, 128)
loop 2 hidden: (2, 64, 128)
exit_probs sum: 1.0000
pool() → (2, 128)
loop loss: X.XXXX ✓ finite
✓ GPU forward pass clean
✓ Base transformer test complete
```

Once that passes, `execution_reasoner.py` is next — it's the first specialist and inherits everything from `STRIDETransformer`, so it's mostly adding input projections and output heads on top of what's already built.
