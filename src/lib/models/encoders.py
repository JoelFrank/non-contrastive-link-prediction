import torch
import torch.nn as nn
from torch_geometric.nn import BatchNorm, GCNConv, LayerNorm, Sequential
from torch_geometric.data import Data
from enum import Enum

# Modelos de codificadores disponibles
class EncoderModel(Enum):
    GCN = 'gcn'

class GCN(nn.Module):
    """Codificador GCN básico.
    Basado en la implementación oficial del codificador BGRL.
    """

    def __init__(
        self,
        layer_sizes,
        batchnorm=False,
        batchnorm_mm=0.99,
        layernorm=False,
        weight_standardization=False,
        use_feat=True,
        n_nodes=0,
        batched=False,
    ):
        super().__init__()

        # batchnorm y layernorm no pueden ser ambos True
        assert batchnorm != layernorm
        assert len(layer_sizes) >= 2
        self.n_layers = len(layer_sizes)
        self.batched = batched
        self.input_size, self.representation_size = layer_sizes[0], layer_sizes[-1]
        self.weight_standardization = weight_standardization

        layers = []
        relus = []
        batchnorms = []

        # Construcción de las capas del modelo
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            if batched:
                # En modo batched, se agregan las capas a listas separadas
                layers.append(GCNConv(in_dim, out_dim))
                relus.append(nn.PReLU())
                if batchnorm:
                    batchnorms.append(BatchNorm(out_dim, momentum=batchnorm_mm))
            else:
                # En modo no batched, se usa Sequential de torch_geometric
                layers.append(
                    (GCNConv(in_dim, out_dim), 'x, edge_index -> x'),
                )

                if batchnorm:
                    layers.append(BatchNorm(out_dim, momentum=batchnorm_mm))
                else:
                    layers.append(LayerNorm(out_dim))

                layers.append(nn.PReLU())

        # Inicialización de las capas según si es batched o no
        if batched:
            # Se usan listas de módulos para cada tipo de capa
            self.convs = nn.ModuleList(layers)
            self.relus = nn.ModuleList(relus)
            self.batchnorms = nn.ModuleList(batchnorms)
        else:
            # Se usa un modelo secuencial para el caso no batched
            self.model = Sequential('x, edge_index', layers)

        self.use_feat = use_feat
        # Si no se usan características, se inicializan embeddings para los nodos
        if not self.use_feat:
            self.node_feats = nn.Embedding(n_nodes, layer_sizes[1])

    def split_forward(self, x, edge_index):
        """Función conveniente para hacer un forward con una matriz de características
        y edge_index por separado sin necesidad de crear un objeto Data.
        """
        return self(Data(x, edge_index))

    def forward(self, data):
        # Forward para el caso no batched
        if not self.batched:
            if self.weight_standardization:
                self.standardize_weights()
            if self.use_feat:
                # Si se usan características, se pasan al modelo
                return self.model(data.x, data.edge_index)
            # Si no, se usan los embeddings de los nodos
            return self.model(self.node_feats.weight.data.clone(), data.edge_index)
        # Forward para el caso batched
        x = data.x
        for i, conv in enumerate(self.convs):
            x = conv(x, data.edge_index)
            x = self.relus[i](x)
            x = self.batchnorms[i](x)
        return x

    def reset_parameters(self):
        # Reinicia los parámetros del modelo
        self.model.reset_parameters()

    def standardize_weights(self):
        # Estandariza los pesos de las capas GCN (excepto la primera)
        skipped_first_conv = False
        for m in self.model.modules():
            if isinstance(m, GCNConv):
                if not skipped_first_conv:
                    skipped_first_conv = True
                    continue
                weight = m.lin.weight.data
                var, mean = torch.var_mean(weight, dim=1, keepdim=True)
                weight = (weight - mean) / (torch.sqrt(var + 1e-5))
                m.lin.weight.data = weight

    def get_node_feats(self):
        # Devuelve los embeddings de nodos si existen
        if hasattr(self, 'node_feats'):
            return self.node_feats
        return None

    @property
    def num_layers(self):
        # Número de capas del modelo
        return self.n_layers

class EncoderZoo:
    """Devuelve un codificador del tipo especificado.
    Lee los flags desde una instancia de absl.FlagValues.
    Ver ../lib/flags.py para los valores y descripciones por defecto.
    """

    # Nota: usamos el valor de los enums ya que los leemos como flags
    models = {EncoderModel.GCN.value: GCN}

    def __init__(self, flags):
        self.flags = flags

    def _init_model(
        self,
        model_class,
        input_size: int,
        use_feat: bool,
        n_nodes: int,
        batched: bool,
        n_feats: int,
    ):
        flags = self.flags
        if model_class == GCN:
            # Inicializa el modelo GCN con los parámetros dados por los flags
            return GCN(
                [input_size] + flags.graph_encoder_layer_dims,
                batchnorm=True,
                use_feat=use_feat,
                n_nodes=n_nodes,
                batched=batched,
            )

    @staticmethod
    def check_model(model_name: str):
        """Verifica si existe un modelo con el nombre dado.
        Lanza un error si no existe.
        """
        if model_name not in EncoderZoo.models:
            raise ValueError(f'Modelo de codificador desconocido: "{model_name}"')
        return True

    def get_model(
        self,
        model_name: str,
        input_size: int,
        use_feat: bool,
        n_nodes: int,
        n_feats: int,
        batched: bool = False,
    ):
        # Devuelve una instancia del modelo solicitado
        EncoderZoo.check_model(model_name)
        return self._init_model(
            EncoderZoo.models[model_name],
            input_size,
            use_feat,
            n_nodes,
            batched=batched,
            n_feats=n_feats,
        )
