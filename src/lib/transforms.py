import copy
import math
import random

import torch
from torch_geometric.transforms import Compose
from torch_geometric.utils import negative_sampling
from torch_geometric.utils.dropout import dropout_adj

# Elimina características de los nodos con probabilidad p.
class DropFeatures:
    r"""Elimina características de los nodos con probabilidad p."""

    def __init__(self, p=None):
        assert p is not None
        assert 0.0 < p < 1.0, (
            'La probabilidad de dropout debe estar entre 0 y 1, pero se obtuvo %.2f' % p
        )
        self.p = p

    def __call__(self, data):
        # Crea una máscara para eliminar columnas de características con probabilidad p
        drop_mask = (
            torch.empty(
                (data.x.size(1),), dtype=torch.float32, device=data.x.device
            ).uniform_(0, 1)
            < self.p
        )
        data.x[:, drop_mask] = 0
        return data

    def __repr__(self):
        return f'{self.__class__.__name__}(p={self.p})'


# Revuelve aleatoriamente las filas de la matriz de características.
class ScrambleFeatures:
    r"""Revuelve aleatoriamente las filas de la matriz de características."""

    def __call__(self, data):
        row_perm = torch.randperm(data.x.size(0))
        data.x = data.x[row_perm, :]
        return data

    def __repr__(self):
        # Nota: self.p no existe aquí, este __repr__ podría causar error.
        return f'{self.__class__.__name__}()'


# Aleatoriza completamente el índice de aristas (edges).
class RandomEdges:
    r"""Aleatoriza completamente el índice de aristas."""

    def __call__(self, data):
        n = data.num_nodes
        # Genera un nuevo edge_index aleatorio con el mismo tamaño
        data.edge_index = torch.randint_like(data.edge_index, n - 1)
        return data

    def __repr__(self):
        return f'{self.__class__.__name__}()'


# Aleatoriza el índice de aristas y el número de aristas en un rango.
class RandomRangeEdges:
    r"""Aleatoriza completamente el índice de aristas y el número de aristas."""

    def __call__(self, data):
        n = data.num_nodes
        n_edges = data.edge_index.size(1)

        # Elige aleatoriamente el número de aristas en un rango de 75% a 125% del original
        n_edges = random.randint(math.ceil(n_edges * 0.75), math.ceil(n_edges * 1.25))
        data.edge_index = torch.randint(
            0, n - 1, (2, n_edges), dtype=data.edge_index.dtype
        ).to(data.edge_index.device)
        return data

    def __repr__(self):
        return f'{self.__class__.__name__}()'


# Elimina aristas con probabilidad p.
class DropEdges:
    r"""Elimina aristas con probabilidad p."""

    def __init__(self, p, force_undirected=False):
        assert p is not None
        assert 0.0 < p < 1.0, (
            'La probabilidad de dropout debe estar entre 0 y 1, pero se obtuvo %.2f' % p
        )

        self.p = p
        self.force_undirected = force_undirected

    def __call__(self, data):
        edge_index = data.edge_index
        edge_attr = data.edge_attr if 'edge_attr' in data else None

        # Aplica dropout a las aristas
        edge_index, edge_attr = dropout_adj(
            edge_index, edge_attr, p=self.p, force_undirected=self.force_undirected
        )

        data.edge_index = edge_index
        if edge_attr is not None:
            data.edge_attr = edge_attr
        return data

    def __repr__(self):
        return f'{self.__class__.__name__}()'


# Añade aristas aleatorias al grafo.
class AddEdges:
    """Realiza la adición aleatoria de aristas."""

    def __init__(self, sample_size_ratio=0.1):
        self.sample_size_ratio = sample_size_ratio

    def __call__(self, data):
        edge_index = data.edge_index
        # Calcula el número de aristas negativas a añadir
        n_samples = round(self.sample_size_ratio * edge_index)
        neg_edges = negative_sampling(
            data.edge_index, num_nodes=data.num_nodes, num_neg_samples=n_samples
        )

        # Añade las aristas negativas al grafo
        edge_index = torch.cat((edge_index, neg_edges))
        data.edge_index = edge_index
        return data

    def __repr__(self):
        return f'{self.__class__.__name__}(sample_size_ratio={self.sample_size_ratio})'


# Aleatoriza completamente la matriz de características (manteniendo el tamaño).
class RandomizeFeatures:
    """Aleatoriza completamente la matriz de características (mantiene el mismo tamaño)."""

    def __init__(self):
        pass

    def __call__(self, data):
        data.x = torch.rand_like(data.x)
        return data

    def __repr__(self):
        return f'{self.__class__.__name__}()'


# Diccionario de transformaciones válidas para augmentaciones.
VALID_TRANSFORMS = dict(
    {
        'standard': ['DropEdges', 'DropFeatures'],
        'all': ['DropEdges', 'DropFeatures'],
        'none': [],
        'drop-edge': ['DropEdges'],
        'drop-feat': ['DropFeatures'],
        'add-edges': ['AddEdges'],
        'add-edges-feat-drop': ['AddEdges', 'DropFeatures'],
    }
)

# Diccionario de transformaciones válidas para corrupciones.
VALID_NEG_TRANSFORMS = dict(
    {
        'heavy-sparsify': ['DropEdges', 'DropFeatures'],
        'randomize-feats': ['RandomizeFeatures'],
        'scramble-feats': ['ScrambleFeatures'],
        'randomize-drop-combo': ['DropEdges', 'RandomizeFeatures'],
        'scramble-drop-combo': ['ScrambleFeatures', 'DropEdges'],
        'scramble-edge-combo': ['ScrambleFeatures', 'RandomEdges'],
        'rand-rand-combo': ['RandomizeFeatures', 'RandomEdges'],
        'rand-rand-rand-combo': ['RandomizeFeatures', 'RandomRangeEdges'],
        'scramble-edge-choice': ['ScrambleFeaturesOrRandomEdges'],
        'scramble-drop-choice': ['ScrambleFeatOrDropEdges'],
        'random-edges': ['RandomEdges'],
        'all-choice': ['AllChoice'],
    }
)

# Permite elegir aleatoriamente entre varias transformaciones al llamar la clase.
class ChooserTransformation:
    """Consiste en múltiples transformaciones.
    Cuando se llama a esta transformación, se selecciona y aplica una de ellas con probabilidad uniforme.
    Esto permite alternar transformaciones durante el entrenamiento del modelo.
    """

    def __init__(self, transformations, transformation_args):
        self.transformations = [
            transformations[i](*transformation_args[i])
            for i in range(len(transformations))
        ]
        self.transformations_str = ',\n'.join([str(x) for x in transformations])

    def __call__(self, data):
        # Selecciona aleatoriamente una transformación y la aplica
        transformation = random.choice(self.transformations)
        return transformation(data)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.transformations_str})'


# Función para componer transformaciones a partir de un nombre de transformación.
def compose_transforms(transform_name, drop_edge_p, drop_feat_p, create_copy=True):
    """Dado un nombre de transformación, retorna el objeto de transformación correspondiente.
    Las transformaciones incluyen tanto aumentaciones como corrupciones.
    El diccionario de aumentaciones válidas está en `VALID_TRANSFORMS`.
    El diccionario de corrupciones válidas está en `VALID_NEG_TRANSFORMS`.
    """

    if transform_name in VALID_TRANSFORMS:
        catalog = VALID_TRANSFORMS[transform_name]
    elif transform_name in VALID_NEG_TRANSFORMS:
        catalog = VALID_NEG_TRANSFORMS[transform_name]
    else:
        raise ValueError('Nombre de transformación desconocido: ', transform_name)

    # Mapeo de nombres de transformaciones a clases y argumentos
    feats = {
        'DropEdges': (DropEdges, [drop_edge_p]),
        'DropFeatures': (DropFeatures, [drop_feat_p]),
        'AddEdges': (AddEdges, []),
        'RandomizeFeatures': (RandomizeFeatures, []),
        'ScrambleFeatures': (ScrambleFeatures, []),
        'RandomEdges': (RandomEdges, []),
        'RandomRangeEdges': (RandomRangeEdges, []),
        'ScrambleFeaturesOrRandomEdges': (
            ChooserTransformation,
            [(ScrambleFeatures, RandomEdges), ([], [])],
        ),
        'ScrambleFeatOrDropEdges': (
            ChooserTransformation,
            [(ScrambleFeatures, DropEdges), ([], [0.95])],
        ),
        'AllChoice': (
            ChooserTransformation,
            [
                (
                    ScrambleFeatures,
                    RandomEdges,
                    RandomizeFeatures,
                    DropFeatures,
                    DropEdges,
                ),
                ([], [], [], [0.95], [0.95]),
            ],
        ),
    }

    transforms = []
    if create_copy:
        # Añade una copia profunda para evitar modificar el objeto original
        transforms.append(copy.deepcopy)

    # Añade las transformaciones seleccionadas según el catálogo
    for transform_name in catalog:
        transform_class, transform_feats = feats[transform_name]
        transforms.append(transform_class(*transform_feats))

    # Devuelve la composición de transformaciones
    return Compose(transforms)
