import torch.nn as nn
    
class CD_Solo(nn.Module):
    def __init__(self, input_dim=23, dropout=0.5, return_features=False):
        super(CD_Solo, self).__init__()
        self.input_dim = input_dim
        self.output_dim = 3
        self.return_features = return_features

        self.clinical_net = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.classifier = nn.Linear(256, self.output_dim)

    def forward(self, x):
        x = x.float()  # Convert input to float
        feat = self.clinical_net(x)
        x = self.classifier(feat)
        if self.return_features: return x, feat
        else: return x