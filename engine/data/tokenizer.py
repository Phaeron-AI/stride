import sys
import torch
from torch import Tensor
from torch.utils.data import Dataset

from pathlib import Path
from typing import Any, Optional, Dict, List, Tuple
from dataclasses import dataclass

engine_root = Path(__file__).parents[1]
if str(engine_root) not in sys.path:
  sys.path.insert(0, str(engine_root))

from config.global_config import STRIDEConfig
from data.execution_tracer import ( 
  ExecutionStep, 
  ExecutionTrace, 
  ExecutionTracer, 
  TraceRecord, 
  ASTFeatureExtractor, 
  TracePipeline
)


class ValueSerializer:
  MAX_VAL_LEN: int = 32

  def serialize(self, value: Any)-> str:
    match value:
      case None:
        result = "None"
      case bool():
        result = str(value)
      case int():
        result = str(value)
      case float():
        result = str(round(value, 4))
      case str():
        result = f"{value}"

      case list() | tuple():
        kind = "list" if isinstance(value, list) else "tuple"
        first = str(value[0]) if value else "empty"
        result = f"{kind}:{len(value)}:{first}"

      case dict():
        first_key = str(next(iter(value))) if value else "empty"
        result = f"dict:{len(value)}:{first_key}"
      
      case _:
        result = type(value).__name__

    return result
  
  def serialize_step(self, step: ExecutionStep)-> str:
    start = f"<|STEP|> line: {step.line_number} depth: {step.depth}"
    parts = [start]

    for k, v in step.local.items():
      parts.append(f"{k}: {self.serialize(v)}")
    
    return " ".join(parts)
  
  def serialize_trace(self, trace: ExecutionTrace)-> str:
    parts = [f"<|BOS|> func: {trace.function_name}"]

    for step in trace.steps:
      parts.append(self.serialize_step(step))
    parts.append("<|EOS|>")

    return " ".join(parts)
  

class TraceVocabulary:
  SPECIAL_TOKENS: Dict[str, int] = {
    "<|PAD|>":  0,
    "<|BOS|>":  1,
    "<|EOS|>":  2,
    "<|SEP|>":  3,
    "<|STEP|>": 4,
    "<|NODE|>": 5,
    "<|UNK|>":  6,
  }

  COMMON_TOKENS: List[str] = [
    # Python keywords
    "def", "return", "for", "while", "if", "else", "elif",
    "import", "from", "class", "try", "except", "raise",
    "with", "as", "in", "not", "and", "or", "is", "None",
    "True", "False", "lambda", "yield", "break", "continue",
    # Common variable name fragments
    "arr", "lst", "result", "output", "value", "count",
    "index", "key", "node", "left", "right", "mid",
    "start", "end", "size", "n", "i", "j", "k", "x", "y",
    # Serializer output prefixes
    "func:", "line:", "depth:", "list:", "tuple:", "dict:",
    # Digits
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    # Type names
    "int", "float", "str", "bool", "NoneType", "object",
    # Punctuation tokens
    ":", ",", ".", "[", "]", "(", ")", "{", "}", "=",
    "+", "-", "*", "/", "<", ">", "!", "_", '"', "**",
  ]

  def __init__(self, vocab_size: int = 32000):
    self.vocab_size = vocab_size

    self.token_to_id: Dict[str, int] = dict(self.SPECIAL_TOKENS)
    self.id_to_token: Dict[int, str] = {v: k for k, v in self.token_to_id.items()}

    self.next_id : int = len(self.SPECIAL_TOKENS)

    for token in self.COMMON_TOKENS:
      self._add_token(token)
  
  def _add_token(self, token: str)-> int:
    if token not in self.token_to_id and self.next_id < self.vocab_size:
      self.token_to_id[token] = self.next_id
      self.id_to_token[self.next_id] = token
      self.next_id += 1
    
    return self.token_to_id.get(token, self.SPECIAL_TOKENS["<|UNK|>"])
  
  def encode(self, text: str)-> List[int]:
    words = text.split()
    return [self._add_token(w) for w in words]
  
  def decode(self, ids: List[int])-> str:
    token = [self.id_to_token.get(i, "<|UNK|>") for i in ids]
    return " ".join(token)
  
  @property
  def pad_id(self) -> int:
    return self.SPECIAL_TOKENS["<|PAD|>"]
 
  @property
  def unk_id(self) -> int:
    return self.SPECIAL_TOKENS["<|UNK|>"]
 
  @property
  def vocab_size_used(self) -> int:
    return self.next_id

class PhysicalStateTokenizer:
  AST_FEATURE_DIM: int = 64

  def __init__(self, config: STRIDEConfig):
    self.cfg = config.data
    self.vocab_cfg = config.vocab
    self.vocab = TraceVocabulary(vocab_size=self.vocab_cfg.vocab_size)
    self.serializer = ValueSerializer()
    self.ast_ext = ASTFeatureExtractor()

    self.max_seq_len = self.cfg.max_seq_len
  
  def encode_trace(self, trace: ExecutionTrace)-> Tuple[List[int], List[float]]:
    text = self.serializer.serialize_trace(trace)
    ids = self.vocab.encode(text)
    ids = ids[:self.max_seq_len]

    mask = [1.0] * len(ids)

    pad_len = self.max_seq_len - len(ids)
    ids  += [self.vocab.pad_id] * pad_len
    mask += [0.0] * pad_len

    return ids, mask
  
  def encode_ast_features(self, trace: ExecutionTrace)-> List[List[float]]:
    line_features = self.ast_ext.extract(trace.source_code)
    line_vectors = [self.ast_ext.to_vector(f) for f in line_features]
    zero_vec = [0.0] * self.AST_FEATURE_DIM
    result = [zero_vec[:] for _ in range(self.max_seq_len)]

    if line_vectors:
      n_lines = len(line_vectors)
      tokens_per_line = max(1, self.max_seq_len // n_lines)

      for i, vec in enumerate(line_vectors):
        start = i * tokens_per_line
        end = min(start + tokens_per_line, self.max_seq_len)
        for j in range(start, end):
          result[j] = vec
    
    return result
  
  def encode_record(self, record: TraceRecord)-> Dict[str, Any]:
    token_ids, attention_mask = self.encode_trace(record.trace)
    ast_features = self.encode_ast_features(record.trace)

    return {
      "token_ids":      token_ids,
      "attention_mask": attention_mask,
      "ast_features":   ast_features,
      "step_labels":    record.step_labels,
      "question":       record.reasoning_question,
      "answer":         record.reasoning_answer,
      "trace_id":       record.trace.trace_id,
    }
  
  def encode_batch(self, records: List[TraceRecord])-> Dict[str, Tensor]:
    B = len(records)

    encoded = [self.encode_record(r) for r in records]

    token_ids    = torch.tensor([e["token_ids"] for e in encoded], dtype=torch.long)
    attention_mask = torch.tensor([e["attention_mask"] for e in encoded], dtype=torch.float)
    ast_features = torch.tensor([e["ast_features"] for e in encoded], dtype=torch.float)

    max_L = max(len(e["step_labels"]) for e in encoded)
    padded = []
    for e in encoded:
      lab = e["step_labels"]
      lab = lab + [-1] * (max_L - len(lab))
      padded.append(lab)
    
    step_labels = torch.tensor(padded, dtype=torch.long)

    assert token_ids.shape      == (B, self.max_seq_len),              \
      f"token_ids: expected {(B, self.max_seq_len)}, got {tuple(token_ids.shape)}"
    assert attention_mask.shape == (B, self.max_seq_len),              \
      f"attention_mask: expected {(B, self.max_seq_len)}, got {tuple(attention_mask.shape)}"
    assert ast_features.shape   == (B, self.max_seq_len, self.AST_FEATURE_DIM), \
      f"ast_features: expected {(B, self.max_seq_len, self.AST_FEATURE_DIM)}, got {tuple(ast_features.shape)}"
    assert step_labels.shape[0] == B,                                  \
      f"step_labels batch dim: expected {B}, got {step_labels.shape[0]}"
    
    return {
      "token_ids":      token_ids,
      "attention_mask": attention_mask,
      "ast_features":   ast_features,
      "step_labels":    step_labels,
    }
  
  def decode_ids(self, ids: List[int])-> str:
    return self.vocab.decode(ids)
  
class TraceDataset(Dataset):
  def __init__(self, records: List[TraceRecord], tokenizer: PhysicalStateTokenizer):
    self.records = records
    self.tokenizer = tokenizer

    self.encoded: List[Dict[str, Any]] = [self.tokenizer.encode_record(r) for r in records]

  def __len__(self)-> int:
    return len(self.records)
  
  def __getitem__(self, index: int) -> Dict[str, Any]:
    return self.encoded[index]
  
  def collate_fn(self, batch: List[Dict[str, Any]])-> Dict[str, Tensor]:
    token_ids = torch.tensor(
      [b["token_ids"] for b in batch], dtype=torch.long
    )
    attention_mask = torch.tensor(
      [b["attention_mask"] for b in batch], dtype=torch.float
    )
    ast_features   = torch.tensor(
      [b["ast_features"] for b in batch], dtype=torch.float
    )
    max_L  = max(len(b["step_labels"]) for b in batch)
    padded = []
    for b in batch:
      lab = b["step_labels"]
      lab = lab + [-1] * (max_L - len(lab))
      padded.append(lab)
    step_labels = torch.tensor(padded, dtype=torch.long)

    return {
      "token_ids":      token_ids,
      "attention_mask": attention_mask,
      "ast_features":   ast_features,
      "step_labels":    step_labels,
    }
  
if __name__ == "__main__":
  config = STRIDEConfig.load()
  config.validate()
 
  # ── Generate traces ──────────────────────
 
  def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
      for j in range(0, n - i - 1):
        if arr[j] > arr[j + 1]:
          arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
 
  def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
      mid = (left + right) // 2
      if arr[mid] == target:
        return mid
      elif arr[mid] < target:
        left = mid + 1
      else:
        right = mid - 1
    return -1
 
  def divide(a, b):
    return a / b
 
  pipeline = TracePipeline(config)
  records = pipeline.run(bubble_sort, [([3, 1, 2],), ([5, 4, 3, 2, 1],)])
  records += pipeline.run(binary_search, [([1, 3, 5, 7, 9], 5), ([1, 3, 5, 7, 9], 4)])
  records += pipeline.run(divide, [(10, 0)])
  print(f"Generated {len(records)} trace records")
 
  # ── Serializer test ──────────────────────
 
  print("\n── Serializer ──────────────────────────")
  ser = ValueSerializer()
  test_values = [42, 3.14, "hello", True, False, [1, 2, 3], {"a": 1}, None, object()]
  for v in test_values:
    print(f"  {repr(v):25s} → {repr(ser.serialize(v))}")
 
  # ── Vocabulary test ──────────────────────
 
  print("\n── Vocabulary ──────────────────────────")
  vocab = TraceVocabulary(vocab_size=32000)
  print(f"vocab used after init: {vocab.vocab_size_used} tokens")
  test_tokens = ["<STEP>", "<BOS>", "def", "unknown_token_xyz"]
  for t in test_tokens:
    print(f"  encode {repr(t):25s} → {vocab.encode(t)}")
 
  # ── Single record encode ─────────────────
 
  print("\n── Single record ───────────────────────")
  tokenizer = PhysicalStateTokenizer(config)
  r = records[0]
 
  token_ids, mask = tokenizer.encode_trace(r.trace)
  ast_feat = tokenizer.encode_ast_features(r.trace)
  encoded = tokenizer.encode_record(r)
 
  print(f"token_ids length: {len(token_ids)}")
  print(f"real tokens: {int(sum(mask))}")
  print(f"pad tokens: {int(len(mask) - sum(mask))}")
  print(f"ast_features shape: ({len(ast_feat)}, {len(ast_feat[0])})")
  print(f"step_labels: {encoded['step_labels'][:5]}...")
  print(f"trace_id: {encoded['trace_id']}")
  print(f"question: {encoded['question']}")
 
  # Decode first 20 real tokens to verify readability
  real_ids = [i for i, m in zip(token_ids, mask) if m > 0][:20]
  print(f"first 20 tokens: {tokenizer.decode_ids(real_ids)}")
 
  # ── Batch encode ─────────────────────────
 
  print("\n── Batch encode ────────────────────────")
  batch = tokenizer.encode_batch(records)
  B = len(records)
  print(f"token_ids: {tuple(batch['token_ids'].shape)} dtype={batch['token_ids'].dtype}")
  print(f"attention_mask: {tuple(batch['attention_mask'].shape)} dtype={batch['attention_mask'].dtype}")
  print(f"ast_features: {tuple(batch['ast_features'].shape)} dtype={batch['ast_features'].dtype}")
  print(f"step_labels: {tuple(batch['step_labels'].shape)} dtype={batch['step_labels'].dtype}")
  print(f"All shapes verified by assertions")
 
  # ── DataLoader test ──────────────────────
 
  print("\n── DataLoader ──────────────────────────")
  import torch.utils.data as tud
 
  dataset = TraceDataset(records, tokenizer)
  dataloader = tud.DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
    collate_fn=dataset.collate_fn,
  )
 
  for i, batch in enumerate(dataloader):
    print(f"batch {i}: token_ids {tuple(batch['token_ids'].shape)}")
    if i >= 2:
      break
 
  print(f"\nDataset size: {len(dataset)} records")
  print(f"Vocab used: {tokenizer.vocab.vocab_size_used} tokens")
 
  print("\nTokenizer test complete")