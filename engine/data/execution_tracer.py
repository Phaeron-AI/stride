import sys
import copy
import ast
import inspect
import hashlib
import textwrap

from typing import Any, Optional, Dict, Tuple, List, Callable
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExecutionStep:
  line_number: int = 0
  event: str = ""
  local: Dict[str, Any] = field(default_factory=dict)
  depth: int = 0


@dataclass
class ExecutionTrace:
  # Fundamental Code Related
  function_name: str = ""
  source_code: str = ""

  # Parameters and arguments
  args: Tuple = field(default_factory=tuple)
  kwargs: Dict = field(default_factory=dict)

  # Intermediate and Outputs
  steps: List[ExecutionStep] = field(default_factory=list)
  final_output: Any = None

  # Error Related
  error: Optional[str] = None
  trace_id: str = ""

  def __post_init__(self):
    if not self.trace_id:
      self.trace_id = _make_trace_id(self.function_name, self.args)

  @property
  def n_steps(self) -> int:
    return len(self.steps)

  @property
  def n_variables(self) -> int:
    # FIX: was referencing self.local instead of step.local
    return max(len(step.local) for step in self.steps) if self.steps else 0

  @property
  def is_valid(self) -> bool:
    return self.error is None


def _make_trace_id(function_name: str, args: Tuple) -> str:
  raw = f"{function_name}_{str(args)}"
  hashed_trace_id = hashlib.md5(raw.encode("utf-8")).hexdigest()
  return hashed_trace_id[:15]


def _is_supported_type(value: Any, supported_types: List[str]) -> bool:
  return type(value).__name__ in supported_types


def _filter_locals(
  locals_dict: Dict[str, Any],
  supported_types: List[str],
  max_variables: int
) -> Dict[str, Any]:
  filtered = {
    k: v for k, v in locals_dict.items()
    if _is_supported_type(v, supported_types)
  }
  if len(filtered) > max_variables:
    filtered = dict(list(filtered.items())[:max_variables])
  return filtered


class ExecutionTracer:
  def __init__(self, config):
    self.cfg = config.data
    self._steps: List[ExecutionStep] = []
    self._depth: int = 0
    # FIX: was self.target_name — inconsistent with self._target_name used in _tracer_function
    self._target_name: str = ""

  def trace(self, function: Callable, *args, **kwargs) -> ExecutionTrace:
    function_name = function.__name__
    source_code = _get_source(function)

    # FIX: was self.steps = [] — should be self._steps
    self._steps = []
    self._depth = 0
    self._target_name = function_name

    output = None
    error = None

    try:
      sys.settrace(self._tracer_function)
      output = function(*args, **kwargs)
    except Exception as e:
      # FIX: was e.with_traceback which is a method, not the message
      error = str(e)
    finally:
      sys.settrace(None)

    self._steps = self._steps[:self.cfg.max_trace_length]

    # FIX: was missing commas between fields — SyntaxError
    return ExecutionTrace(
      function_name=function_name,
      source_code=source_code,
      args=args,
      kwargs=kwargs,
      steps=self._steps,
      final_output=output,
      error=error,
    )

  def _tracer_function(self, frame, event: str, arg: Any) -> Optional[Callable]:
    frame_name = frame.f_code.co_name

    if frame_name != self._target_name:
      return self._tracer_function

    if event == "call":
      self._depth += 1

    elif event == "line":
      filtered = _filter_locals(
        frame.f_locals,
        self.cfg.supported_types,
        self.cfg.max_variables,
      )
      snapshot = copy.deepcopy(filtered)
      step = ExecutionStep(
        line_number=frame.f_lineno,
        event=event,
        local=snapshot,
        depth=self._depth,
      )
      self._steps.append(step)

    elif event == "return":
      self._depth -= 1

    return self._tracer_function

  # FIX: was _trace_batch (private) — made public to match TracePipeline call
  def trace_batch(self, func: Callable, inputs: List[Tuple]) -> List[ExecutionTrace]:
    return [self.trace(func, *args) for args in inputs]


def _get_source(func: Callable) -> str:
  try:
    source = inspect.getsource(func)
    # FIX: added textwrap.dedent and strip — removes leading indentation
    return textwrap.dedent(source).strip()
  except (OSError, TypeError):
    return ""


def _get_ast(source: str) -> Optional[ast.AST]:
  try:
    return ast.parse(source)
  except SyntaxError:
    return None


class ASTFeatureExtractor:
  FEATURE_DIM: int = 64

  # FIX: was indented incorrectly causing IndentationError
  AST_NODES: List[str] = [
    "FunctionDef", "AsyncFunctionDef", "ClassDef", "Return",
    "Delete", "Assign", "AugAssign", "AnnAssign", "For",
    "AsyncFor", "While", "If", "With", "AsyncWith", "Raise",
    "Try", "Assert", "Import", "ImportFrom", "Global",
    "Nonlocal", "Expr", "Pass", "Break", "Continue",
    "BoolOp", "BinOp", "UnaryOp", "Lambda", "IfExp",
    "Dict", "Set", "ListComp", "SetComp", "DictComp",
    "GeneratorExp", "Await", "Yield", "YieldFrom", "Compare",
    "Call", "FormattedValue", "JoinedStr", "Constant", "Attribute",
    "Subscript", "Starred", "Name", "List", "Tuple",
    "Slice", "Load", "Store", "Del", "And",
    "Or", "Add", "Sub", "Mult", "Div",
    "FloorDiv", "Mod", "Pow", "LShift", "RShift",
    "BitOr", "BitXor", "BitAnd", "MatMult",
  ]

  def extract(self, source: str) -> List[Dict[str, Any]]:
    tree = _get_ast(source)
    lines = source.split("\n")
    features = [{} for _ in lines]

    if tree is None:
      return features

    for node in ast.walk(tree):
      lineno = getattr(node, "lineno", None)
      # FIX: was >= which skips valid lines and catches out-of-bounds
      if lineno is not None and lineno <= len(lines):
        features[lineno - 1][type(node).__name__] = 1

    return features

  def to_vector(self, features: Dict[str, Any]) -> List[float]:
    # FIX: was invalid syntax — correct conditional comprehension
    vector = [1.0 if node in features else 0.0 for node in self.AST_NODES]
    assert len(vector) == self.FEATURE_DIM, \
      f"Feature vector length {len(vector)} != FEATURE_DIM {self.FEATURE_DIM}"
    return vector


@dataclass
class TraceRecord:
  trace: ExecutionTrace = field(default_factory=ExecutionTrace)
  ast_features: List[Dict] = field(default_factory=list)
  step_labels: List[int] = field(default_factory=list)
  reasoning_question: str = ""
  reasoning_answer: str = ""


class StepLabelGenerator:
  def generate(self, trace: ExecutionTrace) -> List[int]:
    labels = []
    for i, step in enumerate(trace.steps):
      if trace.error is None:
        labels.append(1)
      else:
        midpoint = len(trace.steps) // 2
        labels.append(1 if i < midpoint else 0)
    return labels


class ReasoningQuestionGenerator:
  TEMPLATES = [
    "What is the value of {var} after line {line}?",
    "How many times does the loop execute in this trace?",
    "What does this function return when called with {args}?",
    "At which line does {var} first appear?",
    "What is the final value of {var}?",
    "Does this function complete without error?",
    "How many variables are in scope at line {line}?",
  ]

  def generate(self, trace: ExecutionTrace) -> List[Tuple[str, str]]:
    pairs = []

    if trace.is_valid:
      question = self.TEMPLATES[2].format(args=trace.args)
      answer = str(trace.final_output)
      pairs.append((question, answer))

    for step in trace.steps[:3]:
      for var, val in step.local.items():
        question = self.TEMPLATES[0].format(var=var, line=step.line_number)
        answer = str(val)
        pairs.append((question, answer))

    question = self.TEMPLATES[5]
    answer = "yes" if trace.is_valid else "no"
    pairs.append((question, answer))

    return pairs


class TracePipeline:
  def __init__(self, config):
    self.tracer = ExecutionTracer(config=config)
    self.ast_ext = ASTFeatureExtractor()
    self.labeler = StepLabelGenerator()
    self.questioner = ReasoningQuestionGenerator()

  def run(self, func: Callable, inputs: List[Tuple]) -> List[TraceRecord]:
    records = []
    # FIX: was self.tracer._trace_batch — now public trace_batch
    traces = self.tracer.trace_batch(func=func, inputs=inputs)

    for trace in traces:
      ast_features = self.ast_ext.extract(trace.source_code)
      step_labels = self.labeler.generate(trace=trace)
      qa_pairs = self.questioner.generate(trace=trace)

      question = qa_pairs[0][0] if qa_pairs else ""
      answer = qa_pairs[0][1] if qa_pairs else ""

      record = TraceRecord(
        trace=trace,
        ast_features=ast_features,
        step_labels=step_labels,
        reasoning_question=question,
        reasoning_answer=answer,
      )
      records.append(record)

    return records


if __name__ == "__main__":
  sys.path.append(str(Path(__file__).parents[2]))
  from config.global_config import STRIDEConfig

  config = STRIDEConfig.load()
  config.validate()

  def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
      for j in range(0, n - i - 1):
        if arr[j] > arr[j + 1]:
          arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

  def factorial(n):
    if n <= 1:
      return 1
    return n * factorial(n - 1)

  def divide(a, b):
    return a / b

  pipeline = TracePipeline(config)

  print("\n── Bubble Sort ──────────────────────────")
  records = pipeline.run(bubble_sort, [([3, 1, 2],), ([5, 4, 3, 2, 1],)])
  for r in records:
    print(f"  trace_id:    {r.trace.trace_id}")
    print(f"  n_steps:     {r.trace.n_steps}")
    print(f"  n_variables: {r.trace.n_variables}")
    print(f"  is_valid:    {r.trace.is_valid}")
    print(f"  output:      {r.trace.final_output}")
    print(f"  labels:      {r.step_labels[:5]}...")
    print(f"  question:    {r.reasoning_question}")
    print(f"  answer:      {r.reasoning_answer}")
    print()

  print("── Factorial ────────────────────────────")
  records = pipeline.run(factorial, [(5,), (10,)])
  for r in records:
    print(f"  n_steps: {r.trace.n_steps}  output: {r.trace.final_output}")

  print("\n── Error Trace ──────────────────────────")
  records = pipeline.run(divide, [(10, 0)])
  for r in records:
    print(f"  is_valid: {r.trace.is_valid}  error: {r.trace.error}")
    print(f"  labels:   {r.step_labels}")

  print("\n── AST Features ─────────────────────────")
  extractor = ASTFeatureExtractor()
  features = extractor.extract(_get_source(bubble_sort))
  vector = extractor.to_vector(features[2]) if len(features) > 2 else []
  print(f"  Lines extracted:     {len(features)}")
  print(f"  Feature vector len:  {len(vector)}")
  print(f"  Non-zero features:   {sum(1 for v in vector if v > 0)}")

  print("\n✓ Execution tracer test complete")