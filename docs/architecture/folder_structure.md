stride/
│
├── main.py
├── .gitignore
├── README.md
├── setup.py
│
├── docs/
│   ├── 00_global_config.md
│   ├── 01_execution_tracer.md
│   ├── 02_tokenizer.md
│   ├── 03_execution_reasoner.md
│   ├── 04_causal_reasoner.md
│   ├── 05_complexity_reasoner.md
│   ├── 06_process_reward_model.md
│   ├── 07_reasoning_graph_builder.md
│   ├── 08_code_generation_head.md
│   ├── 09_syntax_verifier.md
│   ├── 10_test_generator.md
│   ├── 11_component_interfaces.md
│   └── updated/
│       ├── 00_architecture_overview.md
│       ├── 00_global_summary.md
│       ├── 03_execution_reasoner.md
│       ├── 04_causal_reasoner.md
│       ├── 05_complexity_reasoner.md
│       ├── 06_process_reward_model.md
│       ├── 07_reasoning_graph_builder.md
│       ├── 08_code_generation_head.md
│       └── 09_10_verifier_testgen.md
│
└── engine/
    ├── requirements.txt
    │
    ├── config/
    │   ├── global_config.py
    │   ├── components/
    │   │   ├── system.yaml
    │   │   ├── vocab.yaml
    │   │   ├── training.yaml
    │   │   ├── data.yaml
    │   │   ├── execution_reasoner.yaml
    │   │   ├── causal_reasoner.yaml
    │   │   ├── complexity_reasoner.yaml
    │   │   ├── prm.yaml
    │   │   ├── graph_builder.yaml
    │   │   ├── generation.yaml
    │   │   ├── syntax_verifier.yaml
    │   │   └── test_generator.yaml
    │   └── experiments/
    │       └── README.md
    │
    ├── data/
    │   ├── __init__.py
    │   ├── execution_tracer.py       ← ExecutionTracer + helpers  ✓ done
    │   ├── tokenizer.py              ← PhysicalStateTokenizer      ✓ done
    │   ├── vocabulary.py             ← TraceVocabulary             split out
    │   ├── value_serializer.py       ← ValueSerializer             split out
    │   ├── dataset.py                ← TraceDataset                split out
    │   ├── pipeline.py               ← DatasetBuilder              ✓ done
    │   └── cache/                    ← saved train/val/test json
    │
    ├── reasoning/
    │   ├── __init__.py
    │   │
    │   ├── components/               ← building blocks (1 class each)
    │   │   ├── __init__.py
    │   │   ├── rms_norm.py           ← RMSNorm
    │   │   ├── rotary_embedding.py   ← RotaryEmbedding
    │   │   ├── gqa.py                ← GroupedQueryAttention
    │   │   ├── swiglu.py             ← SwiGLUFFN
    │   │   ├── block_attn_res.py     ← BlockAttnRes
    │   │   ├── transformer_layer.py  ← TransformerLayer
    │   │   └── exit_gate.py          ← ExitGate
    │   │
    │   ├── base_transformer.py       ← STRIDETransformer + LoopState
    │   ├── execution_reasoner.py     ← ExecutionReasoner
    │   ├── causal_reasoner.py        ← CausalReasoner
    │   ├── complexity_reasoner.py    ← ComplexityReasoner
    │   └── graph_builder.py          ← ReasoningGraphBuilder
    │
    ├── reward/
    │   ├── __init__.py
    │   └── process_reward_model.py   ← ProcessRewardModel
    │
    ├── generation/
    │   ├── __init__.py
    │   ├── code_generation_head.py   ← CodeGenerationHead
    │   ├── syntax_verifier.py        ← SyntaxVerifier
    │   └── test_generator.py         ← TestGenerator
    │
    ├── training/
    │   ├── __init__.py
    │   ├── trainer.py                ← Trainer (base)
    │   ├── prm_trainer.py            ← PRMTrainer
    │   ├── specialist_trainer.py     ← SpecialistTrainer
    │   └── generation_trainer.py     ← GenerationTrainer
    │
    └── evaluation/
        ├── __init__.py
        └── benchmarks.py             ← BenchmarkRunner