from typing import Optional, Any, Dict, List

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

  def __init__(self, vocab_size: int = 32000)-> None:
    self.vocab_size = vocab_size

    self.token_to_id: Dict[str, int] = dict(self.SPECIAL_TOKENS)
    self.id_to_token: Dict[int, str] = {v: k for k, v in self.token_to_id.items()}
    self.next_id: int = len(self.SPECIAL_TOKENS)

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
    tokens = [self.id_to_token.get(i, "<|UNK|>") for i in ids]
    return " ".join(tokens)
  
  @property
  def pad_id(self) -> int:
    return self.SPECIAL_TOKENS["<|PAD|>"]
 
  @property
  def unk_id(self) -> int:
    return self.SPECIAL_TOKENS["<|UNK|>"]
 
  @property
  def vocab_size_used(self) -> int:
    return self.next_id