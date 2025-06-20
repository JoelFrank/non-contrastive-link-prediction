"""Este archivo entrena varios métodos no contrastivos.
Esto incluye T-BGRL (nuestro método propuesto), BGRL, CCA-SSG y GBT.
El modelo se puede seleccionar con el flag base_model
(por ejemplo, --base_model=triplet para T-BGRL).
"""
import logging
import os
from os import path
import time
import json
from absl import app
from absl import flags
import torch
from torch import nn
import wandb
from lib.data import get_dataset
from lib.models.decoders import DecoderZoo
from lib.models import EncoderZoo
from lib.eval import do_all_eval, do_inductive_eval
from ogb.linkproppred import PygLinkPropPredDataset
from lib.training import (
    perform_bgrl_training,
    perform_cca_ssg_training,
    perform_gbt_training,
    perform_triplet_training,
)
from lib.transforms import VALID_NEG_TRANSFORMS
from lib.split import do_transductive_edge_split, do_node_inductive_edge_split
from lib.utils import (
    is_small_dset,
    merge_multirun_results,
    print_run_num,
)
import lib.flags as FlagHelper

######
# Flags / Parámetros
######
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

FLAGS = flags.FLAGS

# Definir flags compartidos
FlagHelper.define_flags(FlagHelper.ModelGroup.NCL)
flags.DEFINE_enum(
    'base_model', 'bgrl', ['gbt', 'bgrl', 'triplet', 'cca'], 'Qué modelo base usar.'
)
flags.DEFINE_float('mm', 0.99, 'El momentum para el promedio móvil.')
flags.DEFINE_integer('predictor_hidden_size', 512, 'Tamaño oculto del proyector.')

flags.DEFINE_enum(
    'negative_transforms',
    'randomize-feats',
    list(VALID_NEG_TRANSFORMS.keys()),
    'Qué transformaciones negativas de grafo usar (solo para triplet).',
)
flags.DEFINE_bool('eval_only', False, 'Solo evaluar el modelo.')
flags.DEFINE_multi_enum(
    'eval_only_pred_model',
    [],
    ['lr', 'mlp', 'cosine', 'seal', 'prod_lr'],
    'Qué modelos de predicción de enlaces usar (sobrescribe link_pred_model si eval_only es True y esto está configurado)',
)

# Flags relacionados con el batching
flags.DEFINE_bool(
    'batch_graphs',
    False,
    'Si se realiza batching en los grafos. Solo implementado para BGRL y T-BGRL.',
)
flags.DEFINE_integer(
    'graph_batch_size', 1024, 'Número de subgrafos por minibatch.'
)
flags.DEFINE_integer(
    'graph_eval_batch_size',
    128,
    'Número de subgrafos por minibatch en evaluación. Solo si batch_graphs es True.',
)
flags.DEFINE_integer(
    'n_workers',
    0,
    'Número de workers para el dataloader. Solo si batch_graphs es True.',
)
flags.DEFINE_integer(
    'n_batch_neighbors',
    50,
    'Número de vecinos para el minibatching. Solo si batch_graphs es True.',
)

flags.DEFINE_integer('lr_warmup_epochs', 1000, 'Periodo de warmup para la tasa de aprendizaje.')
flags.DEFINE_bool(
    'training_early_stop',
    False,
    'Si se realiza early stopping en la pérdida de entrenamiento',
)
flags.DEFINE_integer(
    'training_early_stop_patience', 50, 'Paciencia para early stopping en entrenamiento'
)

# Flags de corrupción
flags.DEFINE_float(
    'add_edge_ratio_1',
    0.0,
    'Proporción de aristas negativas a muestrear (comparado con aristas positivas existentes) para la red online.',
)
flags.DEFINE_float(
    'add_edge_ratio_2',
    0.0,
    'Proporción de aristas negativas a muestrear (comparado con aristas positivas existentes) para la red target.',
)
flags.DEFINE_float(
    'neg_lambda', 0.5, 'Peso para la cabeza triplet negativa. Entre 0 y 1'
)

# Flags específicos del modelo de predicción de enlaces
flags.DEFINE_bool(
    'save_extra', False, 'Si se guarda información extra para depuración/plotting'
)
flags.DEFINE_bool(
    'dataset_fixed',
    True,
    'Si se corrigió un bug de message-passing vs aristas normales',
)

flags.DEFINE_float('cca_lambda', 0.0, 'Lambda para CCA-SSG')


def get_full_model_name():
    """Devuelve el nombre completo del modelo según los flags."""
    model_prefix = 'I'
    edge_prob_str = f'dep1{FLAGS.drop_edge_p_1}_dfp1{FLAGS.drop_feat_p_1}_dep2{FLAGS.drop_edge_p_2}_dfp2{FLAGS.drop_feat_p_2}'
    if FLAGS.model_name_prefix:
        model_prefix = FLAGS.model_name_prefix + '_' + model_prefix

    if FLAGS.base_model == 'gbt':
        return f'{model_prefix}GBT_{FLAGS.dataset}_lr{FLAGS.lr}_mm{FLAGS.mm}_{edge_prob_str}'
    elif FLAGS.base_model == 'triplet':
        return f'{model_prefix}TBGRL_{FLAGS.dataset}_lr{FLAGS.lr}_mm{FLAGS.mm}_{edge_prob_str}'

    return (
        f'{model_prefix}BGRL_{FLAGS.dataset}_lr{FLAGS.lr}_mm{FLAGS.mm}_{edge_prob_str}'
    )


######
# Main / Principal
######
def main(_):
    log.info('¡Ejecución iniciada!')

    # Sobrescribe el modelo de predicción de enlaces si corresponde
    if FLAGS.eval_only_pred_model and FLAGS.eval_only:
        log.info(
            f'Se sobrescribe el valor actual de eval_only_pred_model ({FLAGS.link_pred_model}) con {FLAGS.eval_only_pred_model}'
        )
        FLAGS.link_pred_model = FLAGS.eval_only_pred_model

    # Establece el directorio de logs si no está definido
    if FLAGS.logdir is None:
        new_logdir = f'./runs/{FLAGS.dataset}'
        log.info(f'No se estableció logdir, usando por defecto {new_logdir}')
        FLAGS.logdir = new_logdir

    # Configura el muestreo negativo trivial según el dataset
    if FLAGS.trivial_neg_sampling == 'auto':
        if FLAGS.dataset == 'ogbl-collab':
            FLAGS.trivial_neg_sampling = 'true'
            log.info(
                f'Se establece trivial_neg_sampling en true porque auto está configurado y el dataset es grande'
            )
        else:
            FLAGS.trivial_neg_sampling = 'false'
            log.info(
                f'Se establece trivial_neg_sampling en true porque auto está configurado y el dataset es pequeño'
            )

    # Inicializa wandb
    wandb.init(
        project=f'fixed-{FLAGS.base_model}-prod',
        config={'model_name': get_full_model_name(), **FLAGS.flag_values_dict()},
    )

    # Usa CUDA si está disponible
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    log.info('Usando {} para entrenamiento.'.format(device))
    enc_zoo = EncoderZoo(FLAGS)
    dec_zoo = DecoderZoo(FLAGS)

    # Verifica modelos válidos
    enc_zoo.check_model(FLAGS.graph_encoder_model)
    valid_models = DecoderZoo.filter_models(FLAGS.link_pred_model)
    log.info(f'Modelos de validación de predicción de enlaces encontrados: {FLAGS.link_pred_model}')
    log.info(f'Usando modelo de encoder: {FLAGS.graph_encoder_model}')

    if wandb.run is None:
        raise ValueError('¡Fallo al inicializar wandb run!')

    # Crea el directorio de salida
    OUTPUT_DIR = os.path.join(FLAGS.logdir, f'{get_full_model_name()}_{wandb.run.id}')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Guarda el archivo de configuración de flags
    with open(
        path.join(OUTPUT_DIR, 'eval_config.cfg' if FLAGS.eval_only else 'config.cfg'),
        "w",
    ) as file:
        file.write(FLAGS.flags_into_string())  # guarda el archivo de configuración

    # Guarda la configuración en JSON
    with open(path.join(OUTPUT_DIR, 'config.json'), 'w') as f:
        json.dump(FLAGS.flag_values_dict(), f)

    # Carga los datos
    st_time = time.time_ns()
    dataset = get_dataset(FLAGS.dataset_dir, FLAGS.dataset)
    data = dataset[0]  # todos los datasets (actualmente) son solo 1 grafo

    small_dataset = is_small_dset(FLAGS.dataset)
    if small_dataset:
        log.info(
            'Se detectó un dataset pequeño, se usarán configuraciones pequeñas para el split inductivo.'
        )

    if isinstance(dataset, PygLinkPropPredDataset):
        raise NotImplementedError()

    # Realiza el split de los datos según el método seleccionado
    if FLAGS.split_method == 'transductive':
        edge_split = do_transductive_edge_split(dataset, FLAGS.split_seed)
        data.edge_index = edge_split['train']['edge'].t()  # type: ignore
        data.to(device)
        training_data = data
    else:  # inductivo
        (
            training_data,
            val_data,
            inference_data,
            data,
            test_edge_bundle,
            negative_samples,
        ) = do_node_inductive_edge_split(
            dataset=dataset, split_seed=FLAGS.split_seed, small_dataset=small_dataset
        )  # type: ignore

    end_time = time.time_ns()
    log.info(f'Se tardó {(end_time - st_time) / 1e9}s en cargar los datos')

    log.info('Dataset {}, {}.'.format(dataset.__class__.__name__, data))

    # Solo mueve los datos si se usa batch completo
    if not FLAGS.batch_graphs:
        training_data = training_data.to(device)

    # Construye las redes
    has_features = True
    input_size = data.x.size(1)  # type: ignore
    representation_size = FLAGS.graph_encoder_layer_dims[-1]

    train_cb = None

    all_results = []
    all_times = []
    total_times = []
    time_bundle = None

    # Entrena y evalúa para cada corrida
    for run_num in range(FLAGS.num_runs):
        print_run_num(run_num)

        if FLAGS.base_model == 'bgrl':
            encoder, representations, time_bundle = perform_bgrl_training(
                data=training_data,
                output_dir=OUTPUT_DIR,
                representation_size=representation_size,
                device=device,
                input_size=input_size,
                has_features=has_features,
                g_zoo=enc_zoo,
                train_cb=train_cb,
                extra_return=FLAGS.save_extra,
            )
            if FLAGS.save_extra:
                predictor = representations
            log.info('¡Entrenamiento finalizado!')
        elif FLAGS.base_model == 'cca':
            time_bundle = None
            encoder, representations, time_bundle = perform_cca_ssg_training(
                data=training_data,
                output_dir=OUTPUT_DIR,
                device=device,
                input_size=input_size,
                has_features=has_features,
                g_zoo=enc_zoo,
            )
            log.info('¡Entrenamiento finalizado!')
        elif FLAGS.base_model == 'gbt':
            encoder, representations, time_bundle = perform_gbt_training(
                training_data, OUTPUT_DIR, device, input_size, has_features, enc_zoo
            )
            # del encoder
            log.info('Entrenamiento finalizado')
        elif FLAGS.base_model == 'triplet':
            encoder, representations, time_bundle = perform_triplet_training(
                data=training_data.to(device),
                output_dir=OUTPUT_DIR,
                representation_size=representation_size,
                device=device,
                input_size=input_size,
                has_features=has_features,
                g_zoo=enc_zoo,
                train_cb=train_cb,
            )
        else:
            raise NotImplementedError()

        # Guarda los tiempos de entrenamiento si corresponde
        if time_bundle is not None:
            (total_time, _, _, times) = time_bundle
            all_times.append(times.tolist())
            total_times.append(int(total_time))

        # Evalúa el modelo según el método de split
        if FLAGS.split_method == 'transductive':
            embeddings = nn.Embedding.from_pretrained(representations, freeze=True)
            results, _ = do_all_eval(
                get_full_model_name(),
                output_dir=OUTPUT_DIR,
                valid_models=valid_models,
                dataset=dataset,
                edge_split=edge_split,
                embeddings=embeddings,
                lp_zoo=dec_zoo,
                wb=wandb,
            )
        else:  # inductivo
            results = do_inductive_eval(
                model_name=get_full_model_name(),
                output_dir=OUTPUT_DIR,
                encoder=encoder,
                valid_models=valid_models,
                train_data=training_data,
                val_data=val_data,
                inference_data=inference_data,
                lp_zoo=dec_zoo,
                device=device,
                test_edge_bundle=test_edge_bundle,
                negative_samples=negative_samples,
                wb=wandb,
                return_extra=FLAGS.save_extra,
            )

        if FLAGS.save_extra:
            nn_model, results = results
        all_results.append(results)

    # Guarda información extra si se solicita
    if FLAGS.save_extra:
        torch.save(
            {
                'nn_model': nn_model.state_dict(),
                'predictor': predictor.state_dict(),
                'encoder': encoder.state_dict(),
            },
            path.join(OUTPUT_DIR, 'extra_data.pt'),
        )
        torch.save(
            (
                training_data,
                val_data,
                inference_data,
                data,
                test_edge_bundle,
                negative_samples,
            ),
            path.join(OUTPUT_DIR, 'data_split.pt'),
        )
    print(all_results)
    agg_results, to_log = merge_multirun_results(all_results)
    wandb.log(to_log)

    # Guarda los tiempos si corresponde
    if time_bundle is not None:
        with open(f'{OUTPUT_DIR}/times.json', 'w') as f:
            json.dump({'all_times': all_times, 'total_times': total_times}, f)

    # Guarda los resultados agregados
    with open(f'{OUTPUT_DIR}/agg_results.json', 'w') as f:
        json.dump(agg_results, f)

    log.info(f'¡Listo! La información de la ejecución se encuentra en {OUTPUT_DIR}')


if __name__ == "__main__":
    app.run(main)
