from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional
import yaml

def _load_yaml(path: str | Path) -> Dict[str, Any]:
  path = Path(path)
  if not path.exists():
    return {}
  with open(path) as f:
    return yaml.safe_load(f) or {}


def _merge(base: Dict, override: Dict) -> Dict:
  result = base.copy()
  for key, value in override.items():
    if key in result and isinstance(result[key], dict) and isinstance(value, dict):
      result[key] = _merge(result[key], value)
    else:
      result[key] = value
  return result


# Path to component yaml files
COMPONENTS_DIR: Path = Path(__file__).parent / "components"

@dataclass
class SystemConfig:
  model_name: str = "STRIDE"
  version: str = "0.1.0"
  precision: str = "bf16"
  device: str = "cuda"
  seed: int = 42

@dataclass
class VocabConfig:
  vocab_size: int = 32000
  pad_token_id: int = 0
  bos_token_id: int = 1
  eos_token_id: int = 2
  sep_token_id: int = 3
  step_token_id: int = 4
  node_token_id: int = 5

@dataclass
class TrainingConfig:
  batch_size: int = 8
  gradient_accumulation_steps: int = 8
  effective_batch_size: int = 0
  learning_rate: float = 3e-4
  warmup_steps: int = 1000
  optimizer: str = "adamw"
  max_grad_norm: float = 1.0

  def __post_init__(self):
    self.effective_batch_size = self.batch_size * self.gradient_accumulation_steps

@dataclass
class DataConfig:
  max_trace_length: int = 512
  max_variables: int = 64
  max_seq_len: int = 512
  token_dim: int = 512
  supported_types: List[str] = field(default_factory=lambda: [
    "int", "float", "double", "str", "bool", "list", "dict", "tuple"
  ])


@dataclass
class ExecutionReasonerConfig:
  d_model: int = 768
  n_heads: int = 12
  d_head: int = 0
  n_layers: int = 24
  d_ff: int = 0
  dropout: float = 0.1
  max_seq_len: int = 512

  def __post_init__(self):
    self.d_head = self.d_model // self.n_heads
    self.d_ff   = 4 * self.d_model
    assert self.d_model % self.n_heads == 0, \
      f"d_model {self.d_model} must be divisible by n_heads {self.n_heads}"


@dataclass
class CausalReasonerConfig:
  d_model: int = 768
  n_heads: int = 12
  d_head: int = 0
  n_layers: int = 24
  d_ff: int = 0
  dropout: float = 0.1
  max_seq_len: int = 512
  ast_feature_dim: int = 64
  edge_types: int = 6

  def __post_init__(self):
    self.d_head = self.d_model // self.n_heads
    self.d_ff   = 4 * self.d_model
    assert self.d_model % self.n_heads == 0, \
      f"d_model {self.d_model} must be divisible by n_heads {self.n_heads}"


@dataclass
class ComplexityReasonerConfig:
  d_model: int = 512
  n_heads: int = 8
  d_head: int  = 0
  n_layers: int = 16
  d_ff: int = 0
  dropout: float = 0.1
  max_seq_len: int = 512
  complexity_classes: int = 9

  def __post_init__(self):
    self.d_head = self.d_model // self.n_heads
    self.d_ff   = 4 * self.d_model
    assert self.d_model % self.n_heads == 0, \
      f"d_model {self.d_model} must be divisible by n_heads {self.n_heads}"


@dataclass
class PRMConfig:
  d_model: int = 512
  n_heads: int = 8
  d_head: int = 0
  n_layers: int = 16
  d_ff: int = 0
  dropout: float = 0.1
  max_step_len: int = 256
  max_graph_nodes: int = 64
  score_dimensions: int = 4
  score_weights: List[float] = field(
    default_factory=lambda: [0.4, 0.3, 0.2, 0.1]
  )
  cross_attn_every: int = 4

  def __post_init__(self):
    self.d_head = self.d_model // self.n_heads
    self.d_ff = 4 * self.d_model
    assert self.d_model % self.n_heads == 0, \
        f"d_model {self.d_model} must be divisible by n_heads {self.n_heads}"
    assert len(self.score_weights) == self.score_dimensions, \
      f"len(score_weights) {len(self.score_weights)} " \
      f"must equal score_dimensions {self.score_dimensions}"
    assert round(sum(self.score_weights), 6) == 1.0, \
      f"score_weights must sum to 1.0, got {sum(self.score_weights)}"

@dataclass
class GraphBuilderConfig:
  d_model: int = 512
  n_heads: int = 8
  d_head: int = 0
  n_layers: int = 8
  d_ff: int = 0
  dropout: float = 0.1
  max_seq_len: int = 512
  max_nodes: int = 64
  tokens_per_node: int = 0
  edge_types: int = 6
  node_types: int = 4

  def __post_init__(self):
    self.d_head = self.d_model // self.n_heads
    self.d_ff = 4 * self.d_model
    self.tokens_per_node = self.max_seq_len // self.max_nodes
    assert self.d_model % self.n_heads == 0, \
      f"d_model {self.d_model} must be divisible by n_heads {self.n_heads}"
    assert self.max_seq_len % self.max_nodes == 0, \
      f"max_seq_len {self.max_seq_len} must be divisible " \
      f"by max_nodes {self.max_nodes}"


@dataclass
class GenerationConfig:
  d_model:                int   = 768
  n_heads:                int   = 12
  d_head:                 int   = 0
  n_layers:               int   = 24
  d_ff:                   int   = 0
  dropout:                float = 0.1
  max_prompt_len:         int   = 256
  max_gen_len:            int   = 1024
  graph_node_dim:         int   = 512
  graph_proj_dim:         int   = 0
  graph_cross_attn_every: int   = 4

  def __post_init__(self):
    self.d_head        = self.d_model // self.n_heads
    self.d_ff          = 4 * self.d_model
    self.graph_proj_dim = self.d_model
    assert self.d_model % self.n_heads == 0, \
      f"d_model {self.d_model} must be divisible by n_heads {self.n_heads}"
    assert self.graph_proj_dim == self.d_model, \
      f"graph_proj_dim must equal d_model for cross attention"

@dataclass
class SyntaxVerifierConfig:
  d_model:              int   = 256
  n_heads:              int   = 4
  d_head:               int   = 0
  n_layers:             int   = 6
  d_ff:                 int   = 0
  dropout:              float = 0.1
  max_seq_len:          int   = 256
  syntax_error_classes: int   = 12
  verify_every:         int   = 16

  def __post_init__(self):
    self.d_head = self.d_model // self.n_heads
    self.d_ff   = 4 * self.d_model
    assert self.d_model % self.n_heads == 0, \
      f"d_model {self.d_model} must be divisible by n_heads {self.n_heads}"

@dataclass
class TestGeneratorConfig:
  d_model:           int   = 512
  n_heads:           int   = 8
  d_head:            int   = 0
  n_layers:          int   = 12
  d_ff:              int   = 0
  dropout:           float = 0.1
  max_code_len:      int   = 512
  max_spec_len:      int   = 256
  max_combined_len:  int   = 0
  max_test_len:      int   = 512
  test_types:        int   = 4
  max_retries:       int   = 3
  execution_timeout: int   = 5
  encoder_layers:    int   = 0
  decoder_layers:    int   = 0

  def __post_init__(self):
    self.d_head          = self.d_model // self.n_heads
    self.d_ff            = 4 * self.d_model
    self.max_combined_len = self.max_code_len + self.max_spec_len
    self.encoder_layers  = self.n_layers // 2
    self.decoder_layers  = self.n_layers // 2
    assert self.d_model % self.n_heads == 0, \
      f"d_model {self.d_model} must be divisible by n_heads {self.n_heads}"
    assert self.encoder_layers + self.decoder_layers == self.n_layers, \
      f"encoder_layers + decoder_layers must equal n_layers"



@dataclass
class STRIDEConfig:
  system:              SystemConfig           = field(default_factory=SystemConfig)
  vocab:               VocabConfig            = field(default_factory=VocabConfig)
  training:            TrainingConfig         = field(default_factory=TrainingConfig)
  data:                DataConfig             = field(default_factory=DataConfig)
  execution_reasoner:  ExecutionReasonerConfig  = field(default_factory=ExecutionReasonerConfig)
  causal_reasoner:     CausalReasonerConfig   = field(default_factory=CausalReasonerConfig)
  complexity_reasoner: ComplexityReasonerConfig = field(default_factory=ComplexityReasonerConfig)
  prm:                 PRMConfig              = field(default_factory=PRMConfig)
  graph_builder:       GraphBuilderConfig     = field(default_factory=GraphBuilderConfig)
  generation:          GenerationConfig       = field(default_factory=GenerationConfig)
  syntax_verifier:     SyntaxVerifierConfig   = field(default_factory=SyntaxVerifierConfig)
  test_generator:      TestGeneratorConfig    = field(default_factory=TestGeneratorConfig)

  @classmethod
  def load(cls, override: Optional[str] = None) -> "STRIDEConfig":
    system              = _load_yaml(COMPONENTS_DIR / "system.yaml")
    vocab               = _load_yaml(COMPONENTS_DIR / "vocab.yaml")
    training            = _load_yaml(COMPONENTS_DIR / "training.yaml")
    data                = _load_yaml(COMPONENTS_DIR / "data.yaml")
    execution_reasoner  = _load_yaml(COMPONENTS_DIR / "execution_reasoner.yaml")
    causal_reasoner     = _load_yaml(COMPONENTS_DIR / "causal_reasoner.yaml")
    complexity_reasoner = _load_yaml(COMPONENTS_DIR / "complexity_reasoner.yaml")
    prm                 = _load_yaml(COMPONENTS_DIR / "prm.yaml")
    graph_builder       = _load_yaml(COMPONENTS_DIR / "graph_builder.yaml")
    generation          = _load_yaml(COMPONENTS_DIR / "generation.yaml")
    syntax_verifier     = _load_yaml(COMPONENTS_DIR / "syntax_verifier.yaml")
    test_generator      = _load_yaml(COMPONENTS_DIR / "test_generator.yaml")

    if override:
      exp = _load_yaml(override)
      component_map = {
        "system":              system,
        "vocab":               vocab,
        "training":            training,
        "data":                data,
        "execution_reasoner":  execution_reasoner,
        "causal_reasoner":     causal_reasoner,
        "complexity_reasoner": complexity_reasoner,
        "prm":                 prm,
        "graph_builder":       graph_builder,
        "generation":          generation,
        "syntax_verifier":     syntax_verifier,
        "test_generator":      test_generator,
      }
      for key, exp_values in exp.items():
        if key in component_map and isinstance(exp_values, dict):
          component_map[key] = _merge(component_map[key], exp_values)

    return cls(
      system              = SystemConfig(**system)                     if system              else SystemConfig(),
      vocab               = VocabConfig(**vocab)                       if vocab               else VocabConfig(),
      training            = TrainingConfig(**training)                 if training            else TrainingConfig(),
      data                = DataConfig(**data)                         if data                else DataConfig(),
      execution_reasoner  = ExecutionReasonerConfig(**execution_reasoner)   if execution_reasoner  else ExecutionReasonerConfig(),
      causal_reasoner     = CausalReasonerConfig(**causal_reasoner)    if causal_reasoner     else CausalReasonerConfig(),
      complexity_reasoner = ComplexityReasonerConfig(**complexity_reasoner) if complexity_reasoner else ComplexityReasonerConfig(),
      prm                 = PRMConfig(**prm)                           if prm                 else PRMConfig(),
      graph_builder       = GraphBuilderConfig(**graph_builder)        if graph_builder       else GraphBuilderConfig(),
      generation          = GenerationConfig(**generation)             if generation          else GenerationConfig(),
      syntax_verifier     = SyntaxVerifierConfig(**syntax_verifier)    if syntax_verifier     else SyntaxVerifierConfig(),
      test_generator      = TestGeneratorConfig(**test_generator)      if test_generator      else TestGeneratorConfig(),
    )

  def validate(self):
    # Interface 02 → 03, 02 → 04
    assert self.data.token_dim > 0, \
      "data.token_dim must be positive"

    # Interface 07 → 06
    assert self.graph_builder.max_nodes == self.prm.max_graph_nodes, \
      f"graph_builder.max_nodes {self.graph_builder.max_nodes} " \
      f"must equal prm.max_graph_nodes {self.prm.max_graph_nodes}"

    # Interface 07 → 08
    assert self.generation.graph_node_dim == self.graph_builder.d_model, \
      f"generation.graph_node_dim {self.generation.graph_node_dim} " \
      f"must equal graph_builder.d_model {self.graph_builder.d_model}"

    # Generation head projection check
    assert self.generation.graph_proj_dim == self.generation.d_model, \
      f"generation.graph_proj_dim {self.generation.graph_proj_dim} " \
      f"must equal generation.d_model {self.generation.d_model}"

    # PRM step length vs syntax verifier window
    assert self.prm.max_step_len == self.syntax_verifier.max_seq_len, \
      f"prm.max_step_len {self.prm.max_step_len} " \
      f"must equal syntax_verifier.max_seq_len {self.syntax_verifier.max_seq_len}"

    # Test generator code input vs generation output
    assert self.test_generator.max_code_len <= self.generation.max_gen_len, \
      f"test_generator.max_code_len {self.test_generator.max_code_len} " \
      f"must be <= generation.max_gen_len {self.generation.max_gen_len}"

    print(f"STRIDEConfig v{self.system.version} validated successfully")


if __name__ == "__main__":
  config = STRIDEConfig.load()
  config.validate()

  print(f"Model: {config.system.model_name}")
  print(f"Version: {config.system.version}")
  print(f"Execution d_ff: {config.execution_reasoner.d_ff}")
  print(f"Effective batch size: {config.training.effective_batch_size}")
  print(f"Tokens per node: {config.graph_builder.tokens_per_node}")
  print(f"Encoder layers: {config.test_generator.encoder_layers}")
  print(f"Max combined len: {config.test_generator.max_combined_len}")
  print(f"Graph proj dim: {config.generation.graph_proj_dim}")