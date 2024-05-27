import torch
from torch import nn


class CCCLoss(nn.Module):
    """Lin's Concordance Correlation Coefficient: https://en.wikipedia.org/wiki/Concordance_correlation_coefficient
    Measures the agreement between two variables

    It is a product of
    - precision (pearson correlation coefficient) and
    - accuracy (closeness to 45 degree line)

    Interpretation
    - rho =  1: perfect agreement
    - rho =  0: no agreement
    - rho = -1: perfect disagreement

    Args:
        eps (float, optional): Avoiding division by zero. Defaults to 1e-8.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes CCC loss

        Args:
            x (Tensor): input tensor with shapes (n, 1);
            y (Tensor): target tensor with shapes (n, 1);

        Returns:
            Tensor: 1 - CCC loss value
        """
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        rho = torch.sum(vx * vy) / (
                torch.sqrt(torch.sum(torch.pow(vx, 2))) * torch.sqrt(torch.sum(torch.pow(vy, 2))) + self.eps)
        x_m = torch.mean(x)
        y_m = torch.mean(y)
        x_s = torch.std(x)
        y_s = torch.std(y)
        ccc = 2 * rho * x_s * y_s / (torch.pow(x_s, 2) + torch.pow(y_s, 2) + torch.pow(x_m - y_m, 2))
        return 1 - ccc

class Batch_CCCloss(nn.Module):
    """Valence-Arousal loss of VA Estimation Challenge
    Computes weighted loss between valence and arousal CCC loses

    Args:
        alpha (float, optional): Weighted coefficient for valence. Defaults to 1.
        beta (float, optional): Weighted coefficient for arousal. Defaults to 1.
        eps (float, optional): Avoiding division by zero. Defaults to 1e-8.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.ccc = CCCLoss(eps=eps)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes VA loss

        Args:
            x (Tensor): input tensor with shapes (batch_size, seq_len, 1);
            y (Tensor): target tensor with shapes (batch_size, seq_len, 1);

        Returns:
            Tensor: Batch CCC loss value
        """
        # resize x to (batch_size * n, 1) and y to (batch_size * n, 1)
        x = x.view(-1, 1)
        y = y.view(-1, 1)
        loss = self.ccc(x[:, 0], y[:, 0])
        return loss


if __name__ == "__main__":
    # test two functions on batch data
    x = torch.randn(10, 32,1)
    y = torch.randn(10, 32,1)
    loss = Batch_CCCloss()
    print(loss(x, y))
    print("---------------")
    loss = CCCLoss()
    print(loss(x, y))
    print("---------------")
