import torch
import torch.nn as nn
import torch.nn.functional as F

class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=12, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        
        self.policy_conv = nn.Conv2d(in_channels=128, out_channels=2, kernel_size=1)
        self.policy_fc = nn.Linear(2 * 8 * 8, 4096) 
        
        self.value_conv = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1)
        self.value_fc1 = nn.Linear(1 * 8 * 8, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        p = F.relu(self.policy_conv(x))
        p = p.view(-1, 2 * 8 * 8)
        policy_out = self.policy_fc(p)
        
        v = F.relu(self.value_conv(x))
        v = v.view(-1, 1 * 8 * 8)
        v = F.relu(self.value_fc1(v))
        value_out = torch.tanh(self.value_fc2(v))
        
        return policy_out, value_out