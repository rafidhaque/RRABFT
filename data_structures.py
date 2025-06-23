# data_structures.py
from dataclasses import dataclass, field
from typing import List
import time

@dataclass
class Transaction:
    """Represents a model update submission."""
    client_id: int
    update_hash: str
    off_chain_pointer: str

@dataclass
class Block:
    """Represents a block on the blockchain."""
    block_id: int
    previous_hash: str
    transactions: List[Transaction]
    proposer_id: int
    timestamp: float = field(default_factory=time.time)
    