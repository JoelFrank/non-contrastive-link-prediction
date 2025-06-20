import copy
import os
import time
import torch
from absl import flags
from torch.optim import AdamW
from tqdm import tqdm
import wandb
import torch.nn.functional as F
import logging
from torch_geometric.loader import NeighborLoader

from .utils import get_time_bundle
from ..scheduler import CosineDecayScheduler
from ..utils import compute_data_representations_only
from ..transforms import compose_transforms
from ..models import MlpPredictor, TripletBgrl

FLAGS = flags.FLAGS
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

def perform_triplet_training(
    data,
    output_dir,
    representation_size,
    device,
    input_size: int,
    has_features: bool,
    g_zoo,
    train_cb=None,
):
    """
    Realiza el entrenamiento Triplet-BGRL (T-BGRL).
    Funciona tanto para configuraciones transductivas como inductivas.
    """

    # Composición de las transformaciones para cada vista del grafo
    transform_1 = compose_transforms(
        FLAGS.graph_transforms,
        drop_edge_p=FLAGS.drop_edge_p_1,
        drop_feat_p=FLAGS.drop_feat_p_1,
    )
    transform_2 = compose_transforms(
        FLAGS.graph_transforms,
        drop_edge_p=FLAGS.drop_edge_p_2,
        drop_feat_p=FLAGS.drop_feat_p_2,
    )
    transform_3 = compose_transforms(
        FLAGS.negative_transforms, drop_edge_p=0.95, drop_feat_p=0.95
    )

    # Inicialización del encoder y predictor
    encoder = g_zoo.get_model(
        FLAGS.graph_encoder_model,
        input_size,
        has_features,
        data.num_nodes,
        n_feats=data.x.size(1),
    )
    predictor = MlpPredictor(
        representation_size,
        representation_size,
        hidden_size=FLAGS.predictor_hidden_size,
    )
    model = TripletBgrl(encoder, predictor, has_features=has_features).to(device)
    neg_lambda = FLAGS.neg_lambda

    # Optimizador
    optimizer = AdamW(
        model.trainable_parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay
    )

    # Schedulers para el learning rate y momentum
    lr_scheduler = CosineDecayScheduler(FLAGS.lr, FLAGS.lr_warmup_epochs, FLAGS.epochs)
    mm_scheduler = CosineDecayScheduler(1 - FLAGS.mm, 0, FLAGS.epochs)

    #####
    # Funciones de entrenamiento y evaluación
    #####
    def full_train(step):
        """Realiza un entrenamiento de T-BGRL en modo full-batch"""
        model.train()

        # Actualiza el learning rate
        lr = lr_scheduler.get(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Actualiza el momentum
        mm = 1 - mm_scheduler.get(step)

        optimizer.zero_grad()

        # Si el dataset no tiene features, se generan aleatoriamente
        if not has_features:
            data.x = encoder.get_node_feats().weight.data.clone().detach()
        # Se generan tres vistas del grafo con diferentes transformaciones
        x1, x2, x3 = transform_1(data), transform_2(data), transform_3(data)

        # Forward pass para obtener representaciones y predicciones
        q1, y2 = model(x1, x2)
        q2, y1 = model(x2, x1)
        neg_y = model.forward_target(x3)

        # Cálculo de similitudes coseno entre las representaciones
        sim1 = F.cosine_similarity(q1, y2.detach()).mean()
        sim2 = F.cosine_similarity(q2, y1.detach()).mean()
        neg_sim1 = F.cosine_similarity(q1, neg_y.detach()).mean()
        neg_sim2 = F.cosine_similarity(q2, neg_y.detach()).mean()
        to_log = {
            'sim1': sim1,
            'sim2': sim2,
            'neg_sim1': neg_sim1,
            'neg_sim2': neg_sim2,
        }

        # Cálculo de la función de pérdida triplet
        loss = neg_lambda * (neg_sim1 + neg_sim2) - (1 - neg_lambda) * (sim1 + sim2)

        loss.backward()

        optimizer.step()
        model.update_target_network(mm)

        # Registro de métricas en wandb
        wandb.log(
            {
                'curr_lr': lr,
                'curr_mm': mm,
                'train_loss': loss,
                'step': step,
                'epoch': epoch,
                **to_log,
            },
            step=step,
        )
        return loss

    def batch_train(loader, epoch):
        """Realiza un entrenamiento de T-BGRL en modo minibatch"""
        model.train()

        # Actualiza el learning rate
        lr = lr_scheduler.get(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Actualiza el momentum
        mm = 1 - mm_scheduler.get(epoch)

        # Itera sobre los batches del loader
        for batch in tqdm(iterable=loader, desc='Batches', leave=False):
            batch = batch.to(device)
            optimizer.zero_grad()

            # Se generan tres vistas del batch
            x1, x2, x3 = transform_1(batch), transform_2(batch), transform_3(batch)

            # Forward pass
            q1, y2 = model(x1, x2)
            q2, y1 = model(x2, x1)
            neg_y = model.forward_target(x3)

            # Cálculo de similitudes coseno
            sim1 = F.cosine_similarity(q1, y2.detach()).mean()
            sim2 = F.cosine_similarity(q2, y1.detach()).mean()
            neg_sim1 = F.cosine_similarity(q1, neg_y.detach()).mean()
            neg_sim2 = F.cosine_similarity(q2, neg_y.detach()).mean()
            to_log = {
                'sim1': sim1,
                'sim2': sim2,
                'neg_sim1': neg_sim1,
                'neg_sim2': neg_sim2,
            }

            # Cálculo de la función de pérdida triplet
            loss = neg_lambda * (neg_sim1 + neg_sim2) - (1 - neg_lambda) * (sim1 + sim2)
            loss.backward()

            # Registro de métricas en wandb
            wandb.log(
                {
                    'curr_lr': lr,
                    'curr_mm': mm,
                    'train_loss': loss,
                    'epoch': epoch,
                    **to_log,
                }
            )

        optimizer.step()
        # Actualiza la red objetivo
        model.update_target_network(mm)

    best_loss = None
    last_update_epoch = 0
    times = []

    # Ciclo principal de entrenamiento
    if FLAGS.batch_graphs:
        # Si se usa entrenamiento por lotes de grafos
        times = []
        train_loader = NeighborLoader(
            data,
            num_neighbors=[
                FLAGS.n_batch_neighbors,
            ]
            * encoder.num_layers,
            batch_size=FLAGS.graph_batch_size,
            shuffle=True,
            num_workers=FLAGS.n_workers,
            pin_memory=True,
        )

        for epoch in tqdm(range(1, FLAGS.epochs + 1)):
            st_time = time.time_ns()
            batch_train(train_loader, epoch - 1)
            elapsed = time.time_ns() - st_time
            times.append(elapsed)
    else:
        # Entrenamiento full-batch
        for epoch in tqdm(range(1, FLAGS.epochs + 1)):
            if train_cb is not None:
                train_cb(epoch - 1, model)
            st_time = time.time_ns()

            train_loss = full_train(epoch - 1)

            elapsed = time.time_ns() - st_time
            times.append(elapsed)

            # Early stopping: guarda el mejor modelo según la pérdida
            if best_loss is None or (best_loss - train_loss >= 0.01):
                best_loss = train_loss
                last_update_epoch = epoch
            elif (
                FLAGS.training_early_stop
                and epoch - last_update_epoch > FLAGS.training_early_stop_patience
            ):
                log.info('¡Early stopping realizado!')
                break

    # Obtiene estadísticas de tiempo de entrenamiento
    time_bundle = get_time_bundle(times)

    # Guarda los pesos del encoder entrenado
    torch.save(
        {'model': model.online_encoder.state_dict()},
        os.path.join(output_dir, f'triplet-{FLAGS.dataset}.pt'),
    )
    encoder = copy.deepcopy(model.online_encoder.eval())
    # Calcula las representaciones finales de los datos
    representations = compute_data_representations_only(
        encoder, data, device, has_features=has_features
    )
    # Guarda las representaciones
    torch.save(
        representations, os.path.join(output_dir, f'triplet-{FLAGS.dataset}-repr.pt')
    )

    return encoder, representations, time_bundle
