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
  def n_steps(self)-> int:
    return len(self.steps)

  @property
  def n_variables(self)-> int:
    return max(len(self.local) for step in self.steps) if self.steps else 0 # type: ignore
  
  @property
  def is_valid(self)-> bool:
    if self.error is None:
      return True
    
    return False
  
def _make_trace_id(function_name: str, args: Tuple)-> str:
  raw = f"{function_name}_{str(args)}"
  hashed_trace_id = hashlib.md5(raw.encode('utf-8')).hexdigest()

  return hashed_trace_id[:15]

def _is_supported_type(value: Any, supported_types: List[str])-> bool:
  return type(value).__name__ in supported_types

def _filter_locals(locals_dict: Dict[str, Any], supported_types: List[str], max_variables: int)-> Dict[str, Any]:
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

    self.target_name: str = ""
  
  def trace(self, function: Callable, *args, **kwargs)-> ExecutionTrace:
    function_name = function.__name__
    source_code = _get_source(function)

    self.steps = []
    self._depth = 0
    self._target_name = function_name

    output = None
    error = None

    try:
      sys.settrace(self._tracer_function)
      output = function(*args, **kwargs)
    except Exception as e:
      error = e.with_traceback
    finally:
      sys.settrace(None)
    
    self._steps = self._steps[:self.cfg.max_trace_length]

    return ExecutionTrace(
      function_name=function_name,
      source_code=source_code,
      args=args,
      kwargs=kwargs,
      steps=self._steps
      final_output=output
      error=error
    )
  
  