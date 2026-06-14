import torch
import torch.nn as nn
import torch.optim as optim
from model import ChessNet

def train_step(model, optimizer, state_batch, target_policies, target_values):
    """
    Performs one step of learning.
    state_batch: The board tensors from the self-play games.
    target_policies: The move probabilities MCTS found.
    target_values: The actual winner of the game (1 for White, -1 for Black).
    """
    model.train()

    optimizer.zero_grad()
    pred_policies, pred_values = model(state_batch) #forward pass
    value_loss_fn = nn.MSELoss()
    loss_v = value_loss_fn(pred_values.squeeze(), target_values)


    policy_loss_fn = nn.CrossEntropyLoss()
    loss_p = policy_loss_fn(pred_policies, target_policies)


    total_loss = loss_v + loss_p

    
    total_loss.backward()
    

    optimizer.step()

    return total_loss.item(), loss_v.item(), loss_p.item()


if __name__ == "__main__":
    print("Initializing Training Pipeline...")
    dummy_model = ChessNet()
    
 
    optimizer = optim.Adam(dummy_model.parameters(), lr=0.001)
    

    dummy_state = torch.rand(1, 12, 8, 8) 
    dummy_target_policy = torch.rand(1, 4096) # Fake MCTS probabilities
    dummy_target_policy = dummy_target_policy / dummy_target_policy.sum() 
    dummy_target_value = torch.tensor([1.0])  # Fake result: White won
    
    loss, v_loss, p_loss = train_step(dummy_model, optimizer, dummy_state, dummy_target_policy, dummy_target_value)
    
    print(f"Pipeline Success! Total Loss: {loss:.4f} (Value: {v_loss:.4f}, Policy: {p_loss:.4f})")