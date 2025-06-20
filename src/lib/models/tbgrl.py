import copy
import torch

class TripletBgrl(torch.nn.Module):
    """Clase Triplet-BGRL.
    Similar a la clase BGRL, pero contiene una función forward_target
    que permite pasar datos adicionales a través de la red objetivo.
    """

    def __init__(self, encoder, predictor, has_features):
        super().__init__()
        # Red en línea (online network)
        self.online_encoder = encoder
        self.predictor = predictor
        self.has_features = has_features

        # Red objetivo (target network)
        self.target_encoder = copy.deepcopy(encoder)

        # Reinicializa los pesos de la red objetivo
        self.target_encoder.reset_parameters()
        # Detiene el gradiente para la red objetivo (no se actualiza durante el entrenamiento)
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        # Enlaza las características si no existen (comparten las mismas características)
        if not self.has_features:
            self.target_encoder.node_feats = self.online_encoder.node_feats

    def trainable_parameters(self):
        r"""Devuelve los parámetros que serán actualizados por un optimizador."""
        # Retorna los parámetros del encoder en línea y del predictor
        return list(self.online_encoder.parameters()) + list(
            self.predictor.parameters()
        )

    @torch.no_grad()
    def update_target_network(self, mm):
        r"""Realiza una actualización por momentum de los pesos de la red objetivo.

        Args:
            mm (float): Momentum usado en la actualización por media móvil.
        """
        assert 0.0 <= mm <= 1.0, (
            "El momentum debe estar entre 0.0 y 1.0, se recibió %.5f" % mm
        )
        # Actualiza los parámetros de la red objetivo usando los de la red en línea
        for param_q, param_k in zip(
            self.online_encoder.parameters(), self.target_encoder.parameters()
        ):
            param_k.data.mul_(mm).add_(param_q.data, alpha=1.0 - mm)

    @torch.no_grad()
    def forward_target(self, target_x):
        """Realiza inferencia en la red objetivo sin información de autograd."""
        # Pasa los datos por la red objetivo y desconecta el gradiente
        return self.target_encoder(target_x).detach()

    def forward(self, online_x, target_x):
        # Pasa los datos por la red en línea (online network)
        online_y = self.online_encoder(online_x)

        # Pasa la salida por el predictor
        online_q = self.predictor(online_y)

        # Pasa los datos por la red objetivo (target network) sin gradiente
        with torch.no_grad():
            target_y = self.target_encoder(target_x).detach()
        # Retorna la predicción de la red en línea y la salida de la red objetivo
        return online_q, target_y
