# simulation.py
import numpy as np
from typing import List

# Import our custom modules
from fl_client import FL_Client
from blockchain_core import Mempool, ReputationSC, DelegateNode, dBFTEngine, Blockchain
from data_structures import Transaction, Block

# In simulation.py
# --- REPLACE THE ENTIRE AGGREGATOR CLASS WITH THIS ---

class Aggregator:
    """Orchestrates the FL process post-consensus."""
    def __init__(self, client_ids: List[int], reputation_contract: ReputationSC):
        # In a real scenario, this would be a Keras/PyTorch model
        self.global_model_weights = np.random.rand(10, 10) # Dummy model weights
        self.reputation_contract = reputation_contract
        self.off_chain_storage = {} # {update_hash: full_update_data}

    def process_finalized_block(self, block: Block):
        """Aggregates models from a finalized block and updates reputations."""
        print("\n--- Aggregator Processing Finalized Block ---")
        if not block.transactions:
            print("Block is empty, no aggregation needed.")
            return

        # 1. Fetch updates from off-chain storage
        updates_to_process = []
        client_ids_in_block = []
        for tx in block.transactions:
            update = self.off_chain_storage.get(tx.update_hash)
            if update is not None:
                updates_to_process.append(update)
                client_ids_in_block.append(tx.client_id)
        
        if not updates_to_process:
            print("No valid updates found in off-chain storage.")
            return

        # 2. NEW: Calculate a ROBUST global update using the median
        robust_global_update = np.median(updates_to_process, axis=0)

        # 3. Score contributions against this robust update and update reputations
        print("Updating reputations based on contribution quality...")
        for i, update in enumerate(updates_to_process):
            client_id = client_ids_in_block[i]
            # Quality score based on cosine similarity to the ROBUST global update
            cos_sim = np.dot(update.flatten(), robust_global_update.flatten()) / (np.linalg.norm(update.flatten()) * np.linalg.norm(robust_global_update.flatten()))
            # Scale score to be in a range, e.g., 0-10
            quality_score = max(0, 10 * cos_sim) 
            self.reputation_contract.update_reputation(client_id, quality_score)
        
        # 4. Update the actual global model with the robust update
        self.global_model_weights += robust_global_update
        print("Global model updated using robust median aggregation.")



    def process_finalized_block(self, block: Block):
        """Aggregates models from a finalized block and updates reputations."""
        print("\n--- Aggregator Processing Finalized Block ---")
        if not block.transactions:
            print("Block is empty, no aggregation needed.")
            return

        # 1. Fetch updates from off-chain storage
        updates_to_process = []
        client_ids_in_block = []
        for tx in block.transactions:
            update = self.off_chain_storage.get(tx.update_hash)
            if update is not None:
                updates_to_process.append(update)
                client_ids_in_block.append(tx.client_id)
        
        if not updates_to_process:
            print("No valid updates found in off-chain storage.")
            return

        # 2. NEW: Calculate a ROBUST global update using the median
        # The median is much more resistant to outliers (our poisoned updates)
        # This serves as a more reliable "ground truth" for scoring
        robust_global_update = np.median(updates_to_process, axis=0)

        # 3. Score contributions against this robust update and update reputations
        print("Updating reputations based on contribution quality...")
        for i, update in enumerate(updates_to_process):
            client_id = client_ids_in_block[i]
            # Quality score based on cosine similarity to the ROBUST global update
            cos_sim = np.dot(update.flatten(), robust_global_update.flatten()) / (np.linalg.norm(update.flatten()) * np.linalg.norm(robust_global_update.flatten()))
            # Scale score to be in a range, e.g., 0-10
            # A negative similarity (malicious update) will result in a score of 0
            quality_score = max(0, 10 * cos_sim) 
            self.reputation_contract.update_reputation(client_id, quality_score)
        
        # 4. Update the actual global model with the robust update
        self.global_model_weights += robust_global_update
        print("Global model updated using robust median aggregation.")



    def update_reputations(self, updates: List[np.ndarray], client_ids: List[int], global_update: np.ndarray):
        """Calculates quality scores and tells the SC to update."""
        print("Updating reputations based on contribution quality...")
        for i, update in enumerate(updates):
            client_id = client_ids[i]
            # Quality score based on cosine similarity to the global update
            # (A simple but effective metric)
            cos_sim = np.dot(update.flatten(), global_update.flatten()) / (np.linalg.norm(update.flatten()) * np.linalg.norm(global_update.flatten()))
            # Scale score to be in a range, e.g., 0-10
            quality_score = max(0, 10 * cos_sim) 
            self.reputation_contract.update_reputation(client_id, quality_score)
            
class SimulationEngine:
    """Main driver for the entire simulation."""
    def __init__(self, num_clients: int, num_delegates: int, percent_malicious: float):
        print("=== Initializing Simulation Engine ===")
        self.num_clients = num_clients
        self.num_delegates = num_delegates
        
        # --- FIX: REARRANGE THE ORDER OF INITIALIZATION ---

        # 1. Create Clients
        client_ids = list(range(num_clients))
        num_malicious = int(num_clients * percent_malicious)
        self.clients = [
            FL_Client(cid, is_malicious=(i < num_malicious)) 
            for i, cid in enumerate(client_ids)
        ]
        print(f"Created {num_clients} clients ({num_malicious} malicious).")

        # 2. Define the list of malicious clients (NOW it's available)
        self.malicious_client_ids = [c.client_id for c in self.clients if c.is_malicious]

        # 3. Create Blockchain and Core Components
        self.mempool = Mempool()
        self.reputation_sc = ReputationSC(client_ids)
        self.blockchain = Blockchain()
        
        delegates = [DelegateNode(d_id, self.reputation_sc) for d_id in range(num_delegates)]
        
        # 4. Create the dBFT Engine (NOW self.malicious_client_ids exists)
        self.dbft_engine = dBFTEngine(delegates, self.mempool, self.malicious_client_ids)
        
        # 5. Create the Aggregator
        self.aggregator = Aggregator(client_ids, self.reputation_sc)

        
    def run_simulation(self, num_rounds: int, tau: float, block_size: int, attack_round: int):
        """The main simulation loop."""
        print(f"\n=== Starting Simulation: {num_rounds} rounds, Rep Threshold (tau)={tau}, Attack Round={attack_round} ===")
        
        for r in range(num_rounds):
            print(f"\n{'='*20} ROUND {r+1}/{num_rounds} {'='*20}")
            
            # 1. Local Training Phase
            print("\n--- Local Training Phase ---")
            for client in self.clients:
                client.set_global_model(self.aggregator.global_model_weights)
                update = client.train_local_epoch()
                transaction, full_update = client.create_update_transaction(update)
                
                # Client submits transaction to mempool and full update to off-chain storage
                self.mempool.add_transaction(transaction)
                self.aggregator.off_chain_storage[transaction.update_hash] = full_update
            
            # 2. Consensus Phase
            last_block_hash = self.blockchain.get_last_block_hash()
            # --- THIS IS THE CORRECTED LINE ---
            finalized_block = self.dbft_engine.run_consensus_round(r + 1, tau, block_size, last_block_hash, attack_round)
            
            # 3. Aggregation Phase
            if finalized_block:
                finalized_block.previous_hash = last_block_hash # link it properly
                self.blockchain.add_block(finalized_block)
                self.aggregator.process_finalized_block(finalized_block)

            # 4. Logging and Reporting for the round
            print("\n--- End of Round Report ---")
            # Print reputations of first 5 clients for demonstration
            for i in range(min(5, self.num_clients)):
                rep = self.reputation_sc.get_reputation(i)
                malicious_status = "(Malicious)" if self.clients[i].is_malicious else "(Honest)"
                print(f"  Client {i} {malicious_status}: Reputation = {rep:.2f}")


if __name__ == "__main__":
    # Simulation Parameters
    NUM_CLIENTS = 10
    NUM_DELEGATES = 7
    PERCENT_MALICIOUS = 0.3 # 30% of clients are malicious
    REPUTATION_THRESHOLD = 3.5 # tau
    NUM_ROUNDS = 20 # Let's run for enough rounds to see reputations drop
    BLOCK_SIZE = 5 # Max transactions per block
    ATTACK_ROUND = 15 # The round where the coordinated attack happens

    # Create and run the simulation
    engine = SimulationEngine(NUM_CLIENTS, NUM_DELEGATES, PERCENT_MALICIOUS)
    engine.run_simulation(NUM_ROUNDS, REPUTATION_THRESHOLD, BLOCK_SIZE, ATTACK_ROUND)