import torch
import torch.nn as nn

class GoldenDropout(nn.Module):
    """
    Golden-ratio dropout: keep ≈61.8 % (1/φ) of the elements,
    scale the survivors by 1/(1/φ) = φ so that the expected value
    is unchanged.
    Use exactly like nn.Dropout:
        model = nn.Sequential(
            nn.Linear(…, …),
            GoldenDropout(),
            nn.ReLU(),
            …
        )
    """
    def __init__(self):
        super().__init__()
        # φ = (1 + √5)/2
        phi = (1 + torch.sqrt(torch.tensor(5.0))) / 2
        self.keep_prob = 1.0 / phi          # ≈0.618
        self.scale     = phi                # = 1 / keep_prob
        
    def forward(self, x):
        if self.training:
            # Bernoulli mask: 1 where we KEEP the entry
            mask = torch.rand_like(x) < self.keep_prob
            # Zero-out pruned entries & rescale survivors
            return x * mask.to(x.dtype) * self.scale
        else:
            return x