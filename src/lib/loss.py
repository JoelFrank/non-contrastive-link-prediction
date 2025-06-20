import torch
from torch import nn
import torch.nn.functional as F

EPS = 1e-15

def barlow_twins_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
) -> torch.Tensor:
    """
    Calcula la pérdida de Barlow Twins sobre dos matrices de entrada.
    Tomado de la implementación oficial de GBT en:
    https://github.com/pbielak/graph-barlow-twins/blob/ec62580aa89bf3f0d20c92e7549031deedc105ab/gssl/loss.py
    """

    batch_size = z_a.size(0)      # Número de muestras en el batch
    feature_dim = z_a.size(1)     # Dimensión de las características
    _lambda = 1 / feature_dim     # Factor de regularización

    # Normalización por batch: centrado y escalado por desviación estándar
    z_a_norm = (z_a - z_a.mean(dim=0)) / (z_a.std(dim=0) + EPS)
    z_b_norm = (z_b - z_b.mean(dim=0)) / (z_b.std(dim=0) + EPS)

    # Matriz de correlación cruzada entre las dos vistas
    c = (z_a_norm.T @ z_b_norm) / batch_size

    # Máscara para seleccionar los elementos fuera de la diagonal
    off_diagonal_mask = ~torch.eye(feature_dim).bool()
    # La pérdida penaliza la diferencia de la diagonal respecto a 1 y los elementos fuera de la diagonal respecto a 0
    loss = (1 - c.diagonal()).pow(2).sum() + _lambda * c[off_diagonal_mask].pow(2).sum()

    return loss

def cca_ssg_loss(z1, z2, cca_lambda, N):
    """
    Calcula la pérdida CCA-SSG.
    Tomado de la implementación oficial de CCA-SSG en:
    https://github.com/hengruizhang98/CCA-SSG/blob/cea6e73962c9f2c863d1abfcdf71a2a31de8f983/main.py#L75
    """

    # Producto cruzado entre las dos vistas
    c = torch.mm(z1.T, z2)
    # Producto interno de cada vista consigo misma
    c1 = torch.mm(z1.T, z1)
    c2 = torch.mm(z2.T, z2)

    # Normalización por el número de muestras
    c = c / N
    c1 = c1 / N
    c2 = c2 / N

    # Pérdida de invariancia: suma negativa de la diagonal de la matriz cruzada
    loss_inv = -torch.diagonal(c).sum()
    # Matriz identidad para comparar con las matrices de autocorrelación
    iden = torch.eye(c.shape[0]).to(z1.device)
    # Pérdida de desacoplamiento para cada vista (deberían ser similares a la identidad)
    loss_dec1 = (iden - c1).pow(2).sum()
    loss_dec2 = (iden - c2).pow(2).sum()

    # Combinación de las pérdidas de invariancia y desacoplamiento
    return (1 - cca_lambda) * loss_inv + cca_lambda * (loss_dec1 + loss_dec2)
