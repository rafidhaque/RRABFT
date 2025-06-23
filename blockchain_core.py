# blockchain_core.py
from collections import deque
from typing import List, Dict
import time
import numpy as np

from data_structures import Transaction, Block

class Mempool:
    """A queue for pending transactions."""
    def __init__(self):
        self.pending_transactions = deque()

    def add_transaction(self, transaction: Transaction):
        self.pending_transactions.append(transaction)

    def get_transactions(self, n: int) -> List[Transaction]:
        """Gets up to n transactions from the queue."""
        batch = []
        count = min(n, len(self.pending_transactions))
        for _ in range(count):
            batch.append(self.pending_transactions.popleft())
        return batch

class ReputationSC:
    """Simulates a reputation smart contract on the blockchain."""
    def __init__(self, client_ids: List[int], initial_reputation: float = 5.0, alpha: float = 0.5):
        self.reputations = {client_id: initial_reputation for client_id in client_ids}
        self.alpha = alpha # EMA smoothing factor
        print(f"ReputationSC initialized for {len(client_ids)} clients with initial score {initial_reputation}.")

    def get_reputation(self, client_id: int) -> float:
        return self.reputations.get(client_id, 0)

    def update_reputation(self, client_id: int, quality_score: float):
        """Updates a client's score using an exponential moving average."""
        old_rep = self.reputations.get(client_id, 0)
        new_rep = self.alpha * old_rep + (1 - self.alpha) * quality_score
        self.reputations[client_id] = new_rep
        # print(f"    Reputation updated for Client {client_id}: {old_rep:.2f} -> {new_rep:.2f} (Quality: {quality_score:.2f})")

class DelegateNode:
    """Represents a single delegate in the dBFT consensus group."""
    def __init__(self, delegate_id: int, reputation_contract: ReputationSC):
        self.delegate_id = delegate_id
        self.reputation_contract = reputation_contract

    def cast_vote(self, block: Block, tau: float) -> bool:
        """Casts a vote based on the RA-dBFT validation logic."""
        if not block.transactions:
            return True # Allow empty blocks

        client_ids = [tx.client_id for tx in block.transactions]
        
        # This is where the core logic of Experiment 1 is measured
        start_time = time.perf_counter()
        
        reps = [self.reputation_contract.get_reputation(cid) for cid in client_ids]
        average_reputation = sum(reps) / len(reps)
        
        vote = average_reputation >= tau
        
        end_time = time.perf_counter()
        validation_time = end_time - start_time
        
        # print(f"    Delegate {self.delegate_id}: Avg Rep = {average_reputation:.2f}, Threshold = {tau}, Vote = {'YES' if vote else 'NO'}")
        # In a real experiment, we'd return (vote, validation_time)
        return vote

class dBFTEngine:
    """Manages the dBFT consensus process."""
    def __init__(self, delegates: List[DelegateNode], mempool: Mempool):
        self.delegates = delegates
        self.mempool = mempool
        self.speaker_idx = 0

    def run_consensus_round(self, tau: float, block_size: int, last_block_hash: str) -> Block | None:
        """Simulates one full round of consensus."""
        print("\n--- Starting Consensus Round ---")
        if not self.mempool.pending_transactions:
            print("Mempool is empty. No block proposed.")
            return None

        # 1. Propose Block
        speaker = self.delegates[self.speaker_idx]
        print(f"Speaker is Delegate {speaker.delegate_id}")
        transactions = self.mempool.get_transactions(block_size)
        
        proposed_block = Block(
            block_id=0, # Will be set by the blockchain
            previous_hash=last_block_hash,
            transactions=transactions,
            proposer_id=speaker.delegate_id
        )

        # 2. Vote on Block
        yes_votes = 0
        for delegate in self.delegates:
            if delegate.cast_vote(proposed_block, tau):
                yes_votes += 1
        
        print(f"Voting Result: {yes_votes} / {len(self.delegates)} YES votes.")

        # 3. Check for Consensus
        # In dBFT, f = floor((n-1)/3), and you need n-f votes. For simplicity, we use 2/3.
        if yes_votes >= (2/3 * len(self.delegates)):
            print("CONSENSUS REACHED. Block is finalized.")
            self.speaker_idx = (self.speaker_idx + 1) % len(self.delegates)
            return proposed_block
        else:
            print("CONSENSUS FAILED. Block is rejected.")
            # Re-add transactions to mempool for next round
            for tx in proposed_block.transactions:
                self.mempool.add_transaction(tx)
            self.speaker_idx = (self.speaker_idx + 1) % len(self.delegates)
            return None

class Blockchain:
    """Simulates the blockchain ledger."""
    def __init__(self):
        self.chain = [self._create_genesis_block()]
        
    def _create_genesis_block(self) -> Block:
        return Block(block_id=0, previous_hash="0", transactions=[], proposer_id=-1)
        
    def add_block(self, block: Block):
        block.block_id = len(self.chain)
        self.chain.append(block)
        print(f"Block {block.block_id} added to the blockchain.")
        
    def get_last_block_hash(self) -> str:
        # In a real system, this would be a proper hash. We simulate it.
        return hashlib.sha256(str(self.chain[-1]).encode()).hexdigest()