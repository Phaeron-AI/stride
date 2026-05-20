# Component 01 — Execution Tracer

## Identity

```
Type:                   Deterministic Tool
Parameters:             None (not a neural model)
Role:                   Captures program state at every execution step
Output:                 Raw execution traces for dataset construction
```

---

## Hyperparameters

```
max_trace_length:       512        maximum execution steps captured
max_variables:          64         maximum variables tracked per step
supported_types:        int, float, str, bool, list, dict, tuple
```

---

## Dimensional Walkthrough

```
INPUT
  function source          string
  function arguments       tuple of any types

PROCESSING
  per execution step       Dict {
                             line_number:  int
                             event:        str  (line / call / return)
                             locals:       Dict[str, Any]
                           }
  full trace               List[Dict]     length T (variable)

SERIALIZED OUTPUT
  shape                    (T, V)
    T = number of execution steps, max 512
    V = number of variables tracked, max 64

EXAMPLE
  input function           10 lines, 3 variables
  execution steps          50
  output shape             (50, 3)

DATASET RECORD SHAPE
  code tokens              string
  trace                    (T, V)
  step labels              (T,)      1 = correct, 0 = incorrect
  final output             Any       ground truth from interpreter
```

---

## Outputs Fed To

```
→  Component 02   Tokenizer
     raw trace (T, V) converted to token sequence
```

---

## Notes

```
- Ground truth is always the Python interpreter
- Non-deterministic functions (random, time) are seeded before tracing
- Recursive functions tracked with call stack depth counter
- Trace truncated at max_trace_length if exceeded
- Variables not in supported_types are type-tagged and skipped
```