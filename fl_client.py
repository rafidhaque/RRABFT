# fl_client.py
import numpy as np
import hashlib
from data_structures import Transaction

class PoisoningAttacker:
    """A collection of methods for poisoning attacks."""
    @staticmethod
    def sign_flip(update: np.ndarray) -> np.ndarray:
        """A simple model poisoning attack that flips the sign of all weights."""
        print(f"    - Applying sign-flip attack.")
        return -1 * update

class FL_Client:
    """Simulates a single federated learning client."""
    def __init__(self, client_id: int, is_malicious: bool):
        self.client_id = client_id
        self.is_malicious = is_malicious
        # In a real implementation, this would be a Keras/PyTorch model
        self.model_weights = None

    def set_global_model(self, global_model_weights: np.ndarray):
        """Updates the local model with the latest global weights."""
        self.model_weights = global_model_weights

    def train_local_epoch(self) -> np.ndarray:
        """
        Simulates one epoch of local training.
        In a real scenario, this would involve actual model training.
        Here, we simulate it by generating a random update.
        """
        print(f"  Client {self.client_id}: Training local model...")
        # Simulate training by creating a small random change to the weights
        # This is where real training logic would go.
        update = np.random.randn(*self.model_weights.shape) * 0.1
        
        if self.is_malicious:
            update = PoisoningAttacker.sign_flip(update)
            
        return update

    def create_update_transaction(self, update: np.ndarray) -> (Transaction, np.ndarray):
        """Hashes the update and creates a transaction object."""
        update_str = str(update.tobytes())
        update_hash = hashlib.sha256(update_str.encode()).hexdigest()
        
        transaction = Transaction(
            client_id=self.client_id,
            update_hash=update_hash,
            off_chain_pointer=f"ipfs://{update_hash}"
        )
        return transaction, update