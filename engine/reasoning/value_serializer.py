from typing import Any

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
        result = value
      case list() | tuple():
        kind  = "list" if isinstance(value, list) else "tuple"
        first = str(value[0]) if value else "empty"
        result = f"{kind}:{len(value)}:{first}"
      case dict():
        first_key = str(next(iter(value))) if value else "empty"
        result = f"dict:{len(value)}:{first_key}"
      case _:
        result = type(value).__name__
 
    return result[:self.MAX_VAL_LEN]
  
  def serialize_step(self, step)-> str:
    parts = [f"<|STEP|> line:{step.line_number} depth:{step.depth}"]

    for k, v in step.local.items():
      parts.append(f"{k}:{self.serialize(v)}")
    return " ".join(parts)
  
  def serialize_trace(self, trace)-> str:
    parts = [f"<|BOS|> func:{trace.function_name}"]
    for step in trace.steps:
      parts.append(self.serialize_step(step))
    parts.append("<|EOS|>")
    return " ".join(parts)