from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class LossWeightingStrategy(ABC, nn.Module):
    """Base class for loss weighting strategies."""
    
    @abstractmethod
    def weight_losses(self, ntxent_loss: torch.Tensor, da_loss: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Weight the NTXent and Deep Alignment losses.
        
        Args:
            ntxent_loss: The NTXent loss value
            da_loss: The Deep Alignment loss value
            
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Weighted NTXent and DA losses
        """
        pass

class UncertaintyWeighting(LossWeightingStrategy):
    """
    Implements uncertainty weighting as described in Kendall et al. 
    'Multi-Task Learning Using Uncertainty to Weigh Losses'
    """
    def __init__(self, init_log_sigma_ntxent: float = 0.0, init_log_sigma_da: float = 0.0):
        super().__init__()
        # Initialize log_sigma parameters to 0 so that exp(0)=1 initially
        self.log_sigma_ntxent = nn.Parameter(torch.tensor(init_log_sigma_ntxent))
        self.log_sigma_da = nn.Parameter(torch.tensor(init_log_sigma_da))
    
    def weight_losses(self, ntxent_loss: torch.Tensor, da_loss: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weighted_ntxent = torch.exp(-self.log_sigma_ntxent) * ntxent_loss + self.log_sigma_ntxent
        weighted_da = torch.exp(-self.log_sigma_da) * da_loss + self.log_sigma_da
        return weighted_ntxent, weighted_da

    def get_log_vars(self) -> dict:
        """Returns current log variance values for logging."""
        return {
            "log_sigma_ntxent": self.log_sigma_ntxent.item(),
            "log_sigma_da": self.log_sigma_da.item()
        }

class ConstantWeighting(LossWeightingStrategy):
    """Simple constant weighting between losses."""
    def __init__(self, lambda_ntxent: float = 1.0,   lambda_da: float = 1.0):
        super().__init__()
        self.lambda_ntxent = lambda_ntxent
        self.lambda_da = lambda_da
    
    def weight_losses(self, ntxent_loss: torch.Tensor, da_loss: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.lambda_ntxent * ntxent_loss, self.lambda_da * da_loss

    def get_log_vars(self) -> dict:
        """Returns empty dict as there are no learnable parameters."""
        return {} 