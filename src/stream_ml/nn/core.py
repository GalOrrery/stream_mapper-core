"""Neural Network."""

from __future__ import annotations

# STDLIB
from typing import TYPE_CHECKING

# THIRD-PARTY
import torch as xp
import torch.nn as nn
from torch import sigmoid

if TYPE_CHECKING:
    # THIRD-PARTY
    from torch import Tensor

    # LOCAL
    from stream_ml._typing import ParsT

__all__: list[str] = []


# TODO! make this the model class?!
class Net(nn.Module):  # type: ignore[misc]
    """Net."""

    param_names = ("fraction", "phi2_mu", "phi2_sigma")

    def __init__(self, sigma_upper_limit: float = 0.3, fraction_upper_limit: float = 0.45) -> None:
        # Initializes methods from nn.Module, which is necessary to use all of
        # the functions that we
        # want:https://www.educative.io/edpresso/what-is-super-in-python
        super().__init__()

        # The priors. # TODO! implement in the Stream/Background models
        self.sigma_upper_limit = sigma_upper_limit
        self.fraction_upper_limit = fraction_upper_limit

        # Define the layers of the neural network:
        # Total: 1 (phi) -> 3 (fraction, mean, sigma)
        self.fc1 = nn.Linear(1, 50)  # layer 1: 1 node -> 50 nodes
        self.fc2 = nn.Linear(50, 50)  # layer 2: 50 node -> 50 nodes
        self.fc3 = nn.Linear(50, 3)  # layer 3: 50 node -> 3 nodes

    @classmethod
    def unpack_pars(cls, p_arr: Tensor) -> ParsT:
        """Unpack parameters into a dictionary.

        This function takes a parameter array and unpacks it into a dictionary
        with the parameter names as keys.

        Parameters
        ----------
        p_arr : Tensor
            Parameter array.

        Returns
        -------
        ParsT
        """
        p_dict = {}
        for i, name in enumerate(cls.param_names):
            p_dict[name] = p_arr[:, i].view(-1, 1)
        return p_dict

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Input.

        Returns
        -------
        Tensor
            fraction, mean, sigma
        """
        x = xp.tanh(self.fc1(x))
        x = xp.tanh(self.fc2(x))
        pred = self.fc3(x)

        # TODO: Implement the priors in the model
        fraction = sigmoid(pred[:, 2]) * self.fraction_upper_limit
        mean = pred[:, 0]
        sigma = sigmoid(pred[:, 1]) * self.sigma_upper_limit

        return xp.vstack([fraction, mean, sigma]).T
