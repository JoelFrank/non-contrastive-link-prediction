import logging
import torch
from torch_geometric.utils import to_networkx
from torch.nn.functional import one_hot
from absl import flags
import pandas as pd

from .models import GraceEncoder

log = logging.getLogger(__name__)
FLAGS = flags.FLAGS
SMALL_DATASETS = set(['cora', 'citeseer'])
# Usado para formatear la salida
SHORT_DIVIDER = '=' * 10
LONG_DIVIDER_STR = '=' * 30

def print_run_num(run_num):
    # Imprime separadores y el número de ejecución actual en el log
    log.info(LONG_DIVIDER_STR)
    log.info(LONG_DIVIDER_STR)
    log.info(SHORT_DIVIDER + f'  Run #{run_num}  ' + SHORT_DIVIDER)
    log.info(LONG_DIVIDER_STR)
    log.info(LONG_DIVIDER_STR)

def add_node_feats(data, device, type='degree'):
    # Añade características a los nodos basadas en el grado (número de conexiones)
    assert type == 'degree'

    G = to_networkx(data)  # Convierte el grafo a formato networkx
    degrees = torch.tensor([v for (_, v) in G.degree()])  # Obtiene el grado de cada nodo
    data.x = one_hot(degrees).to(device).float()  # Codifica el grado como one-hot y lo asigna como características
    return data

def keywise_agg(dicts):
    # Calcula la media y desviación estándar para cada clave en una lista de diccionarios
    df = pd.DataFrame(dicts)
    mean_dict = df.mean().to_dict()
    std_dict = df.std().to_dict()
    return mean_dict, std_dict

def keywise_prepend(d, prefix):
    # Añade un prefijo a cada clave de un diccionario
    out = {}
    for k, v in d.items():
        out[prefix + k] = v
    return out

def is_small_dset(dset):
    # Verifica si el dataset es pequeño (por nombre)
    return dset in SMALL_DATASETS

def merge_multirun_results(all_results):
    """Une resultados de múltiples ejecuciones en un solo diccionario.
    """
    runs = zip(*all_results)
    agg_results = []
    val_mean = test_mean = None

    for run_group in runs:
        group_type = run_group[0]['type']
        val_res = [run['val'] for run in run_group]
        test_res = [run['test'] for run in run_group]

        val_mean, val_std = keywise_agg(val_res)
        test_mean, test_std = keywise_agg(test_res)
        agg_results.append(
            {
                'type': group_type,
                'val_mean': val_mean,
                'val_std': val_std,
                'test_mean': test_mean,
                'test_std': test_std,
            }
        )

    assert val_mean is not None
    assert test_mean is not None
    return agg_results, {
        **keywise_prepend(val_mean, 'val_mean_'),
        **keywise_prepend(test_mean, 'test_mean_'),
    }

def compute_representations_only(
    net, dataset, device, has_features=True, feature_type='degree'
):
    """Pre-calcula las representaciones para todo el dataset.
    No incluye las etiquetas de los nodos.

    Retorna:
        torch.Tensor: Representaciones
    """
    net.eval()  # Pone el modelo en modo evaluación
    reps = []

    for data in dataset:
        # forward
        data = data.to(device)
        if not has_features:
            if data.x is not None:
                log.warn('[WARNING] node features overidden in Data object')
            data.x = net.get_node_feats().weight.data  # Usa embeddings aprendidos como características
        elif data.x is None:
            data = add_node_feats(data, device=device, type=feature_type)  # Añade características si no existen

        with torch.no_grad():
            if isinstance(net, GraceEncoder):
                reps.append(net(data.x, data.edge_index))  # Para modelos GraceEncoder
            else:
                reps.append(net(data))  # Para otros modelos

    reps = torch.cat(reps, dim=0)
    return reps

def compute_data_representations_only(net, data, device, has_features=True):
    r"""Pre-calcula las representaciones para todo el dataset.
    No incluye las etiquetas de los nodos.

    Retorna:
        torch.Tensor: Representaciones
    """
    net.eval()
    reps = []

    if not has_features:
        if data.x is not None:
            log.warn('[WARNING] features overidden in adj matrix')
        data.x = net.get_node_feats().weight.data  # Usa embeddings aprendidos como características

    with torch.no_grad():
        reps.append(net(data))

    reps = torch.cat(reps, dim=0).to(device)
    return reps
