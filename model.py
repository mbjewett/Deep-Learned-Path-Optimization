import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleQNetwork(nn.Module):

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        super(SimpleQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
class DeepQNetworkLSTM(nn.Module):

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        super(DeepQNetworkLSTM, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.ln1 = nn.LayerNorm(state_size)
        self.lstm1 = nn.LSTM(state_size, 32)
        
        self.ln2 = nn.LayerNorm(32)
        self.fc1 = nn.Linear(32, 64)
        
        self.drop1 = nn.Dropout(0.05)
        
        self.ln3 = nn.LayerNorm(64)
        self.fc2 = nn.Linear(64, 64)
        
        self.drop2 = nn.Dropout(0.1)
        
        self.ln4 = nn.LayerNorm(64)
        self.fc3 = nn.Linear(64, 16)
        
        self.ln5 = nn.LayerNorm(16)
        self.lstm2 = nn.LSTM(16, action_size)
        

    def forward(self, state):
        x = state
        
        x = self.ln1(x)
        x, y = self.lstm1(x)
        
        x = self.ln2(x)
        x = F.relu(self.fc1(x))
        
        x = self.drop1(x)
        
        x = self.ln3(x)
        x = F.relu(self.fc2(x))
        
        x = self.drop2(x)
        
        x = self.ln4(x)
        x = torch.tanh(self.fc3(x))
        
        x = self.ln5(x)
        x, y = self.lstm2(x)
        
        actions = x
        return actions
    

class DeepQNetwork(nn.Module):

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        super(DeepQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.bn1 = nn.BatchNorm1d(state_size)
        self.fc0 = nn.Linear(state_size, 32)
        
        self.bn2 = nn.BatchNorm1d(32)
        self.fc1 = nn.Linear(32, 64)
        
        self.drop1 = nn.Dropout(0.05)
        
        self.bn3 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 64)
        
        self.drop2 = nn.Dropout(0.1)
        
        self.bn4 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 16)
        
        self.bn5 = nn.BatchNorm1d(16)
        self.fc4 = nn.Linear(16, action_size)
        
    
    def forward(self, state):
        x = state
        
        x = self.bn1(x)
        x = F.relu(self.fc0(x))
        
        #x = self.bn2(x)
        x = F.relu(self.fc1(x))
        
        #x = self.drop1(x)
        
        #x = self.bn3(x)
        x = F.relu(self.fc2(x))
        
        #x = self.drop2(x)
        
        #x = self.bn4(x)
        x = F.relu(self.fc3(x))
        
        #x = self.bn5(x)
        x = torch.tanh(self.fc4(x))
        
        actions = x
        return actions
