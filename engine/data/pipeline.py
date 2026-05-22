import sys
import json
import random
import torch
import torch.utils.data as tud

from pathlib import Path
from typing import Any, Optional, Dict, List, Tuple, Callable
from dataclasses import dataclass, field

engine_root = Path(__file__).parents[1]
if str(engine_root) not in sys.path:
  sys.path.insert(0, str(engine_root))

from config.global_config import STRIDEConfig
from data.execution_tracer import (
  ExecutionTrace,
  TraceRecord,
  TracePipeline,
)
from data.tokenizer import (
  PhysicalStateTokenizer,
  TraceDataset,
)


# ─────────────────────────────────────────
# FUNCTION CORPUS
# ─────────────────────────────────────────

class FunctionCorpus:

  @staticmethod
  def sorting() -> List[Tuple[Callable, List[Tuple]]]:

    def bubble_sort(arr):
      n = len(arr)
      for i in range(n):
        for j in range(0, n - i - 1):
          if arr[j] > arr[j + 1]:
            arr[j], arr[j + 1] = arr[j + 1], arr[j]
      return arr

    def selection_sort(arr):
      n = len(arr)
      for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
          if arr[j] < arr[min_idx]:
            min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
      return arr

    def insertion_sort(arr):
      for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
          arr[j + 1] = arr[j]
          j -= 1
        arr[j + 1] = key
      return arr

    def merge_sort(arr):
      if len(arr) <= 1:
        return arr
      mid   = len(arr) // 2
      left  = merge_sort(arr[:mid])
      right = merge_sort(arr[mid:])
      result = []
      i = j = 0
      while i < len(left) and j < len(right):
        if left[i] <= right[j]:
          result.append(left[i])
          i += 1
        else:
          result.append(right[j])
          j += 1
      result.extend(left[i:])
      result.extend(right[j:])
      return result

    inputs_small  = [([3, 1, 2],), ([1, 2, 3],), ([3, 2, 1],), ([1],), ([2, 1],)]
    inputs_medium = [([5, 4, 3, 2, 1],), ([1, 3, 5, 2, 4],), ([9, 7, 5, 3, 1],)]

    return [
      (bubble_sort,    inputs_small + inputs_medium),
      (selection_sort, inputs_small + inputs_medium),
      (insertion_sort, inputs_small + inputs_medium),
      (merge_sort,     inputs_small + inputs_medium),
    ]

  @staticmethod
  def searching() -> List[Tuple[Callable, List[Tuple]]]:

    def linear_search(arr, target):
      for i, val in enumerate(arr):
        if val == target:
          return i
      return -1

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

    def find_max(arr):
      if not arr:
        return None
      max_val = arr[0]
      for val in arr[1:]:
        if val > max_val:
          max_val = val
      return max_val

    arr_sorted   = [1, 3, 5, 7, 9, 11, 13]
    arr_unsorted = [4, 2, 9, 1, 7]

    # FIX: inputs_search used directly for both — preserves arr_unsorted cases
    inputs_search = [
      (arr_sorted,   7), (arr_sorted,   4),
      (arr_unsorted, 9), (arr_unsorted, 6),
    ]
    inputs_max = [(arr_sorted,), (arr_unsorted,), ([1],), ([],)]

    return [
      (linear_search, inputs_search),
      (binary_search, inputs_search),   # FIX: was losing arr_unsorted cases
      (find_max,      inputs_max),
    ]

  @staticmethod
  def math_and_recursion() -> List[Tuple[Callable, List[Tuple]]]:

    def factorial(n):
      if n <= 1:
        return 1
      return n * factorial(n - 1)

    def fibonacci(n):
      if n <= 1:
        return n
      a, b = 0, 1
      for _ in range(2, n + 1):
        a, b = b, a + b
      return b

    def gcd(a, b):
      while b:
        a, b = b, a % b
      return a

    def is_prime(n):
      if n < 2:
        return False
      for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
          return False
      return True

    def power(base, exp):
      result = 1
      for _ in range(exp):
        result *= base
      return result

    return [
      (factorial, [(0,), (1,), (5,), (10,)]),
      (fibonacci, [(0,), (1,), (5,), (10,), (15,)]),
      (gcd,       [(12, 8), (17, 5), (100, 75), (1, 1)]),
      (is_prime,  [(2,), (3,), (4,), (17,), (100,)]),
      (power,     [(2, 8), (3, 4), (5, 3), (1, 10)]),
    ]

  @staticmethod
  def string_operations() -> List[Tuple[Callable, List[Tuple]]]:

    def reverse_string(s):
      result = ""
      for char in s:
        result = char + result
      return result

    def count_vowels(s):
      vowels = "aeiouAEIOU"
      count = 0
      for char in s:
        if char in vowels:
          count += 1
      return count

    def is_palindrome(s):
      s = s.lower()
      return s == s[::-1]

    def word_count(s):
      if not s.strip():
        return 0
      return len(s.split())

    inputs_str = [("hello",), ("",), ("a",), ("hello world",)]

    return [
      (reverse_string, inputs_str),
      (count_vowels,   inputs_str),
      (is_palindrome,  [("racecar",), ("hello",), ("",), ("a",), ("Aba",)]),
      (word_count,     inputs_str + [("  ",)]),
    ]

  @staticmethod
  def error_cases() -> List[Tuple[Callable, List[Tuple]]]:

    def divide(a, b):
      return a / b

    def access_index(arr, idx):
      return arr[idx]

    def parse_int(s):
      return int(s)

    return [
      (divide,       [(10, 2), (10, 0), (0, 5), (7, 0)]),
      (access_index, [([1, 2, 3], 1), ([1, 2, 3], 5), ([], 0)]),
      (parse_int,    [("42",), ("abc",), ("0",), ("",)]),
    ]

  @classmethod
  def all(cls) -> List[Tuple[Callable, List[Tuple]]]:
    return (
      cls.sorting()            +
      cls.searching()          +
      cls.math_and_recursion() +
      cls.string_operations()  +
      cls.error_cases()
    )


# ─────────────────────────────────────────
# DATASET STATS
# ─────────────────────────────────────────

@dataclass
class DatasetStats:
  total_records:   int   = 0
  valid_records:   int   = 0
  error_records:   int   = 0
  total_steps:     int   = 0
  avg_steps:       float = 0.0
  vocab_size_used: int   = 0
  n_functions:     int   = 0


# ─────────────────────────────────────────
# DATASET BUILDER
# ─────────────────────────────────────────

class DatasetBuilder:

  def __init__(self, config: STRIDEConfig):
    self.config    = config
    self.pipeline  = TracePipeline(config)
    self.tokenizer = PhysicalStateTokenizer(config)

  def build(
    self,
    corpus:      List[Tuple[Callable, List[Tuple]]],
    train_ratio: float = 0.8,
    val_ratio:   float = 0.1,
    seed:        int   = 42,
  ) -> Dict[str, TraceDataset]:

    print(f"Generating traces from {len(corpus)} functions...")
    all_records = self._generate_records(corpus)
    print(f"Generated {len(all_records)} records total")

    random.seed(seed)
    random.shuffle(all_records)

    n_total = len(all_records)
    n_train = int(n_total * train_ratio)
    n_val   = int(n_total * val_ratio)

    train_records = all_records[:n_train]
    val_records   = all_records[n_train:n_train + n_val]
    test_records  = all_records[n_train + n_val:]

    print(
      f"Split: {len(train_records)} train / "
      f"{len(val_records)} val / "
      f"{len(test_records)} test"
    )

    return {
      "train": TraceDataset(train_records, self.tokenizer),
      "val":   TraceDataset(val_records,   self.tokenizer),
      "test":  TraceDataset(test_records,  self.tokenizer),
    }

  def _generate_records(
    self,
    corpus: List[Tuple[Callable, List[Tuple]]],
  ) -> List[TraceRecord]:

    all_records = []

    for func, inputs in corpus:
      try:
        records = self.pipeline.run(func, inputs)
        all_records.extend(records)
        print(f"  {func.__name__:<20s} → {len(records)} records")
      except Exception as e:
        print(f"  Warning: {func.__name__} failed: {e}")
        continue

    return all_records

  def compute_stats(
    self,
    records: List[TraceRecord],
  ) -> DatasetStats:

    total = len(records)
    valid = sum(1 for r in records if r.trace.is_valid)
    steps = sum(r.trace.n_steps for r in records)

    return DatasetStats(
      total_records   = total,
      valid_records   = valid,
      error_records   = total - valid,
      total_steps     = steps,
      avg_steps       = steps / total if total > 0 else 0.0,
      vocab_size_used = self.tokenizer.vocab.vocab_size_used,
      n_functions     = len(set(r.trace.function_name for r in records)),
    )

  def save(
    self,
    datasets:   Dict[str, TraceDataset],
    output_dir: str | Path,
  ) -> None:

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, dataset in datasets.items():
      path = output_dir / f"{split_name}.json"
      with open(path, "w") as f:
        json.dump(dataset.encoded, f)
      print(f"  Saved {split_name}: {len(dataset)} records → {path}")

  @staticmethod
  def load(
    input_dir: str | Path,
    config:    STRIDEConfig,
  ) -> Dict[str, TraceDataset]:

    input_dir = Path(input_dir)
    tokenizer = PhysicalStateTokenizer(config)
    datasets  = {}

    for split_name in ["train", "val", "test"]:
      path = input_dir / f"{split_name}.json"

      if not path.exists():
        print(f"  Warning: {path} not found, skipping")
        continue

      with open(path) as f:
        encoded = json.load(f)

      # Bypass re-encoding — inject saved encoded directly
      dataset         = TraceDataset([], tokenizer)
      dataset.encoded = encoded
      datasets[split_name] = dataset

      print(f"  Loaded {split_name}: {len(dataset)} records ← {path}")

    return datasets


# ─────────────────────────────────────────
# DATALOADER FACTORY
# ─────────────────────────────────────────

def make_dataloaders(
  datasets:    Dict[str, TraceDataset],
  batch_size:  int = 8,
  num_workers: int = 0,
) -> Dict[str, tud.DataLoader]:

  loaders = {}

  for split_name, dataset in datasets.items():
    shuffle    = (split_name == "train")
    pin_memory = torch.cuda.is_available()

    loaders[split_name] = tud.DataLoader(
      dataset,
      batch_size  = batch_size,
      shuffle     = shuffle,
      num_workers = num_workers,
      collate_fn  = dataset.collate_fn,
      pin_memory  = pin_memory,
    )

  return loaders


# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────

if __name__ == "__main__":
  config = STRIDEConfig.load()
  config.validate()

  builder = DatasetBuilder(config)

  # ── Build corpus ─────────────────────────

  print("\n── Building corpus ─────────────────────")
  corpus = FunctionCorpus.all()
  print(f"Corpus: {len(corpus)} functions")

  datasets = builder.build(corpus)

  # ── Stats ────────────────────────────────

  print("\n── Dataset stats ───────────────────────")
  for split, ds in datasets.items():
    print(f"  {split:<8s}: {len(ds)} records")

  train_ds  = datasets["train"]
  all_enc   = train_ds.encoded
  avg_steps = sum(len(e["step_labels"]) for e in all_enc) / max(len(all_enc), 1)
  print(f"  train avg steps: {avg_steps:.1f}")
  print(f"  vocab used:      {builder.tokenizer.vocab.vocab_size_used}")

  # ── Save and reload ──────────────────────

  print("\n── Save / load ─────────────────────────")
  save_dir = Path(__file__).parent / "cache"
  builder.save(datasets, save_dir)

  reloaded = DatasetBuilder.load(save_dir, config)
  for split, ds in reloaded.items():
    print(f"  Reloaded {split}: {len(ds)} records")

  orig_ids     = datasets["train"].encoded[0]["token_ids"]
  reloaded_ids = reloaded["train"].encoded[0]["token_ids"]
  assert orig_ids == reloaded_ids, "Save/load mismatch — token_ids differ"
  print("  ✓ Save/load roundtrip verified")

  # ── DataLoaders ──────────────────────────

  print("\n── DataLoaders ─────────────────────────")
  loaders = make_dataloaders(datasets, batch_size=config.training.batch_size)
  for split, loader in loaders.items():
    batch = next(iter(loader))
    print(
      f"  {split:<8s}: {len(loader)} batches · "
      f"token_ids {tuple(batch['token_ids'].shape)}"
    )

  print("\n✓ Pipeline test complete")