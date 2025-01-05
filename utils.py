import os
import argparse
import random
import psutil
import yaml
import logging
# from tensorboardX import SummaryWriter
import wandb

import numpy as np
import torch
import torch.nn as nn
from torch import optim as optim
from tqdm import tqdm
import pandas as pd
from scipy.spatial import distance
from sklearn.metrics.pairwise import pairwise_distances
import dgl
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from datasets.st_loading_utils import load_DLPFC, load_mHypothalamus, create_dictionary_mnn
from datasets.data_proc import Cal_Spatial_Net, simple_impute
import scanpy as sc
import anndata
import scipy

# logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


def get_adjacency_matrix(A, k=6, metric='euclidean'):
    distances = pairwise_distances(A, metric=metric)
    n_samples = A.shape[0]
    adjacency_matrix = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        indices = np.argsort(distances[i])[1:k + 1]  # Exclude the sample itself
        adjacency_matrix[i, indices] = 1

    return adjacency_matrix


def fill_missing_features(node_features, adjacency_matrix):
    filled_features = np.copy(node_features)  # Create a copy of the original node features

    num_nodes = node_features.shape[0]

    for node in range(num_nodes):
        neighbors = np.where(adjacency_matrix[node] != 0)[0]  # Find neighbors of the current node

        if len(neighbors) > 0:
            neighbor_features = node_features[neighbors]  # Get features of neighbors
            avg_features = np.mean(neighbor_features, axis=0)  # Compute average features
            zero_indices = np.where(node_features[node] == 0)[0]  # Find zero feature indices

            if len(zero_indices) > 0:
                filled_features[node, zero_indices] = avg_features[zero_indices]  # Fill zero features with average

    return filled_features

def pair_data_points(obs_names, data, labels, min_distance):
    n = len(data)
    paired_indices = np.zeros(n, dtype=int)
    tree = KDTree(data)

    for i in range(n):
        label = labels[i]
        candidate_indices = np.where(labels != label)[0]
        distances, indices = tree.query(data[i], k=len(candidate_indices))
        valid_indices = candidate_indices[distances > min_distance]

        if valid_indices.size > 0:
            paired_indices[i] = np.random.choice(valid_indices)
        else:
            print('No validation')
            paired_indices[i] = np.random.choice(candidate_indices)

    return obs_names[paired_indices]



def triplet_loss_train(model, graph, x, ad_concat, args, optimizer):
    target_nodes = torch.arange(x.shape[0], device=x.device, dtype=torch.long)
    epoch_iter = tqdm(range(args.max_epoch_triplet))

    """a list of precomputed pi, it has the same length as the iter_comb"""

    # neg_count_avg = 0
    # pos_count_avg = 0
    # neg_dist_count_avg = np.zeros(7)
    # pos_dist_count_avg = np.zeros(7)
    # total_count = 0
    loss_list = []

    # for epoch in epoch_iter:
    #     with torch.no_grad():
    #         z = model.embed(graph, x)

    #     if epoch % 100 == 0 or epoch == 500:
    #         # neg_count = 0
    #         # pos_count = 0
    #         # neg_dist_count = np.zeros(7)
    #         # pos_dist_count = np.zeros(7)

    #         print('Update spot triplets at epoch ' + str(epoch))

    #         ############################# original triplet method #############################

    #         # anchor_ind = []
    #         positive_ind = []
    #         negative_ind = []
    #         # mnn_dict = create_dictionary_mnn(ad_concat, use_rep="MG", batch_name='batch_name', k=50, iter_comb=None)
    #         # for batch_pair in mnn_dict.keys():  # pairwise compare for multiple batches
    #         #     batchname_list = ad_concat.obs['batch_name'][mnn_dict[batch_pair].keys()]
    #         #     cellname_by_batch_dict = dict()
    #         #     for batch_id in range(len(section_ids)):
    #         #         cellname_by_batch_dict[section_ids[batch_id]] = ad_concat.obs_names[
    #         #             ad_concat.obs['batch_name'] == section_ids[batch_id]].values
    #         #
    #         #     anchor_list = []
    #         #     positive_list = []
    #         #     negative_list = []
    #         #     for anchor in mnn_dict[batch_pair].keys():
    #         #         total_count += 1
    #         #         anchor_list.append(anchor)
    #         #         # positive_spot = mnn_dict[batch_pair][anchor][0]  # select the first positive spot
    #         #         positive_spot = ad_concat.obs_names[ad_concat.obs['positive_neighbor']]
    #         #         positive_list.append(positive_spot)
    #         #         section_size = len(cellname_by_batch_dict[batchname_list[anchor]])
    #         #         negative_spot = cellname_by_batch_dict[batchname_list[anchor]][np.random.randint(section_size)]
    #         #         negative_list.append(negative_spot)
    #         #
    #         #         pos_count += (ad_concat.obs['original_clusters'].loc[anchor] != ad_concat.obs['original_clusters'].loc[
    #         #             positive_spot])
    #         #         pos_dist_count[
    #         #             abs(ad_concat.obs['original_clusters'].loc[anchor] - ad_concat.obs['original_clusters'].loc[
    #         #                 positive_spot])] += 1
    #         #         neg_count += (ad_concat.obs['original_clusters'].loc[anchor] == ad_concat.obs['original_clusters'].loc[
    #         #             negative_spot])
    #         #         neg_dist_count[abs(
    #         #             ad_concat.obs['original_clusters'].loc[anchor] - ad_concat.obs['original_clusters'].loc[
    #         #                 negative_spot])] += 1
    #         #
    #         #     batch_as_dict = dict(zip(list(ad_concat.obs_names), range(0, ad_concat.shape[0])))
    #         #     anchor_ind = np.append(anchor_ind, list(map(lambda _: batch_as_dict[_], anchor_list)))
    #         #     positive_ind = np.append(positive_ind, list(map(lambda _: batch_as_dict[_], positive_list)))
    #         #     negative_ind = np.append(negative_ind, list(map(lambda _: batch_as_dict[_], negative_list)))
    #         # pos_count_avg += pos_count
    #         # neg_count_avg += neg_count
    #         # pos_dist_count_avg[:] += pos_dist _count[:]
    #         # neg_dist_count_avg[:] += neg_dist_count[:]

    #         ############################# spatial triplet method #############################

    #         anchor_ind = np.arange(len(ad_concat.obs_names))
    #         positive_list = ad_concat.obs['positive_neighbor'].to_list()

    #         slice_name1, slice_name2 = ad_concat.obs['batch_name'].cat.categories
    #         slice_data1 = np.vstack([ad_concat[ad_concat.obs['batch_name'] == slice_name1].obs['align_row'].values,
    #                                  ad_concat[ad_concat.obs['batch_name'] == slice_name1].obs['align_col'].values]).T
    #         slice_data2 = np.vstack([ad_concat[ad_concat.obs['batch_name'] == slice_name2].obs['align_row'].values,
    #                                  ad_concat[ad_concat.obs['batch_name'] == slice_name2].obs['align_col'].values]).T
    #         slice_label1 = ad_concat[ad_concat.obs['batch_name'] == slice_name1].obs['original_clusters'].values
    #         slice_label2 = ad_concat[ad_concat.obs['batch_name'] == slice_name2].obs['original_clusters'].values
    #         slice_obs_names1 = ad_concat[ad_concat.obs['batch_name'] == slice_name1].obs_names
    #         slice_obs_names2 = ad_concat[ad_concat.obs['batch_name'] == slice_name2].obs_names
    #         neg_ind1 = pair_data_points(slice_obs_names1, slice_data1, slice_label1, 3000)
    #         neg_ind2 = pair_data_points(slice_obs_names2, slice_data2, slice_label2, 3000)

    #         negative_list = np.hstack([neg_ind1, neg_ind2])
    #         batch_as_dict = dict(zip(list(ad_concat.obs_names), range(0, ad_concat.shape[0])))
    #         negative_ind = np.append(negative_ind, list(map(lambda _: batch_as_dict[_], negative_list)))
    #         positive_ind = np.append(positive_ind, list(map(lambda _: batch_as_dict[_], positive_list)))

    #     anchor_arr = z[anchor_ind, ]
    #     positive_arr = z[positive_ind, ]
    #     negative_arr = z[negative_ind, ]

    #     triplet_loss = torch.nn.TripletMarginLoss(margin=1, p=2, reduction='mean')
    #     tri_output = triplet_loss(anchor_arr, positive_arr, negative_arr)
    #     _loss = model(graph, x, targets=target_nodes)

    #     loss = _loss + tri_output
    #     if epoch % 100 == 0 or epoch == 500:
    #         loss_list.append(round(tri_output.item(), 3))

    #     loss.backward()
    #     optimizer.step()

    #     epoch_iter.set_description(
    #         f"# Epoch {epoch}: train_loss: {loss.item():.4f} recon_loss: {_loss.item():.4f} triplet_loss: {tri_output:.4f}")

    #     model.train()
    #     optimizer.zero_grad()

    # # neg_count_avg /= total_count
    # # pos_count_avg /= total_count
    # # pos_dist_count_avg /= total_count
    # # neg_dist_count_avg /= total_count

    # with torch.no_grad():
    #     z = model.embed(graph, x)


    adata_concat_ = ad_concat
    section_ids = args.section_ids
    for epoch in epoch_iter:
        if epoch % 100 == 0:
            mnn_dict = create_dictionary_mnn(adata_concat_, use_rep="MG", batch_name='batch_name', k=50, iter_comb=None)
            anchor_ind = []
            positive_ind = []
            negative_ind = []
            for batch_pair in mnn_dict.keys():  # pairwise compare for multiple batches
                batchname_list = adata_concat_.obs['batch_name'][mnn_dict[batch_pair].keys()]
                #             print("before add KNN pairs, len(mnn_dict[batch_pair]):",
                #                   sum(adata_new.obs['batch_name'].isin(batchname_list.unique())), len(mnn_dict[batch_pair]))

                cellname_by_batch_dict = dict()
                for batch_id in range(len(section_ids)):
                    cellname_by_batch_dict[section_ids[batch_id]] = adata_concat_.obs_names[
                        adata_concat_.obs['batch_name'] == section_ids[batch_id]].values

                anchor_list = []
                positive_list = []
                negative_list = []
                for anchor in mnn_dict[batch_pair].keys():
                    anchor_list.append(anchor)
                    ## np.random.choice(mnn_dict[batch_pair][anchor])
                    positive_spot = mnn_dict[batch_pair][anchor][0]  # select the first positive spot
                    positive_list.append(positive_spot)
                    section_size = len(cellname_by_batch_dict[batchname_list[anchor]])
                    negative_list.append(
                        cellname_by_batch_dict[batchname_list[anchor]][np.random.randint(section_size)])

                batch_as_dict = dict(zip(list(adata_concat_.obs_names), range(0, adata_concat_.shape[0])))
                anchor_ind = np.append(anchor_ind, list(map(lambda _: batch_as_dict[_], anchor_list)))
                positive_ind = np.append(positive_ind, list(map(lambda _: batch_as_dict[_], positive_list)))
                negative_ind = np.append(negative_ind, list(map(lambda _: batch_as_dict[_], negative_list)))
        model.train()
        optimizer.zero_grad()

        _loss = model(graph, x, targets=target_nodes)
        with torch.no_grad():
            z = model.embed(graph, x)

        anchor_arr = z[anchor_ind,]
        positive_arr = z[positive_ind,]
        negative_arr = z[negative_ind,]

        triplet_loss = torch.nn.TripletMarginLoss(margin=1, p=2, reduction='mean')
        tri_output = triplet_loss(anchor_arr, positive_arr, negative_arr)

        loss = _loss + tri_output
        loss.backward()
        optimizer.step()

        # if scheduler is not None:
        #     scheduler.step()
        # loss_dict = {"loss": loss.item()}
        epoch_iter.set_description(f"# Epoch {epoch}: train_loss: {loss.item():.4f}")
        
        # if logger is not None:
        #     # loss_dict["lr"] = get_current_lr(optimizer)
        #     # logger.log(loss_dict, step=epoch)
        #     logger.log({'epoch': epoch+1, 'ratio': r, 'layer-wise acc': lw_acc[0], 'loss':loss.item()})

    with torch.no_grad():
        z = model.embed(graph, x)

    # return z, neg_count_avg, pos_count_avg, neg_dist_count_avg, pos_dist_count_avg
    return z, loss_list


# def model_train(model, graph, optimizer, x, args):
#     target_nodes = torch.arange(x.shape[0], device=x.device, dtype=torch.long)
#     epoch_iter = tqdm(range(args.max_epoch))

#     for epoch in epoch_iter:
#         model.train()
#         loss = model(graph, x, targets=target_nodes)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         epoch_iter.set_description(f"# Epoch {epoch}: train_loss: {loss.item():.4f}")

#         if epoch % 100 == 0:
#             logging.info(f"# Epoch {epoch}: train_loss: {loss.item():.4f}")

#     with torch.no_grad():
#         z = model.embed(graph, x)

#     return model, z
def model_train(model, ad, graph, optimizer, x, args):
    sec_id = ad.obs['batch_name'].cat.categories
    target_nodes = torch.arange(x.shape[0], device=x.device, dtype=torch.long)
    epoch_iter = tqdm(range(args.max_epoch))
    # loss_ari_obs = np.zeros((args.max_epoch // obs_freq, 5))
    for epoch in epoch_iter:
        model.train()
        loss = model(graph, x, targets=target_nodes)
        # with torch.no_grad():
        #     z = model.embed(graph, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_iter.set_description(f"# Epoch {epoch}: train_loss: {loss.item():.4f}")
        # if epoch % obs_freq == 0:
        #     ad_concat = ad.copy()
        #     ad_concat.obs.reset_index(inplace=True, drop=False)
        #     ari, ari_1, ari2 = mclust_ari(ad.obs['original_clusters'], z, ad_concat, sec_id)
        #     loss_ari_obs[epoch // obs_freq] = [ari, ari_1, ari2, round(loss.item(), 3), 0]
        if epoch % 100 == 0:
            logging.info(f"# Epoch {epoch}: train_loss: {loss.item():.4f}")

    with torch.no_grad():
        z = model.embed(graph, x)

    return model, z


def get_pairs(data):
    pairs = [[data[0], data[1]],
             [data[1], data[2]],
             [data[2], data[3]]]
    return pairs


def find_nearest_pairs(crd1, crd2):
    distances = distance.cdist(crd1, crd2, 'euclidean')
    nearest_indices1 = np.argmin(distances, axis=1)
    nearest_indices2 = np.argmin(distances, axis=0)

    return [nearest_indices1, nearest_indices2]


def accuracy(y_pred, y_true):
    y_true = y_true.squeeze().long()
    preds = y_pred.max(1)[1].type_as(y_true)
    correct = preds.eq(y_true).double()
    correct = correct.sum().item()
    return correct / len(y_true)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True


def get_current_lr(optimizer):
    return optimizer.state_dict()["param_groups"][0]["lr"]


def build_args_ST():
    parser = argparse.ArgumentParser(description="GAT")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--dataset", type=str, default="DLPFC")
    parser.add_argument("--st_data_dir", type=str, default="./")
    parser.add_argument("--hl_dir", type=str, default="./")

    parser.add_argument("--section_ids", type=str, help="a list of slice name strings sep by comma, with no spacing")
    parser.add_argument("--num_class", type=int, default=7)
    parser.add_argument("--hvgs", type=int, default=5000)
        
    parser.add_argument("--device", type=int, default=2)
    parser.add_argument("--max_epoch", type=int, default=500,
                        help="number of training epochs")
    parser.add_argument("--max_epoch_triplet", type=int, default=500,
                        help="number of training epochs for triplet loss")
    parser.add_argument("--warmup_steps", type=int, default=-1)

    parser.add_argument("--num_heads", type=int, default=1,
                        help="number of hidden attention heads")
    parser.add_argument("--num_out_heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="number of hidden layers")
    parser.add_argument("--num_dec_layers", type=int, default=2)
    parser.add_argument("--num_remasking", type=int, default=3)  # K views as in paper
    parser.add_argument("--num_hidden", type=str, default="1024,64",
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--in_drop", type=float, default=.1,
                        help="input feature dropout")
    parser.add_argument("--attn_drop", type=float, default=.05,
                        help="attention dropout")
    parser.add_argument("--norm", type=str, default=None)
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0,
                        help="weight decay")
    parser.add_argument("--negative_slope", type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument("--activation", type=str, default="elu")
    parser.add_argument("--mask_rate", type=float, default=0.5)  # mask rate for input node features
    parser.add_argument("--remask_rate", type=float, default=0.5)  # mask rate for node enc features
    parser.add_argument("--remask_method", type=str, default="random")
    parser.add_argument("--mask_type", type=str, default="mask",
                        help="`mask` or `drop`")
    parser.add_argument("--mask_method", type=str, default="random")
    parser.add_argument("--drop_edge_rate", type=float, default=0.0)

    parser.add_argument("--encoder", type=str, default="gat")
    parser.add_argument("--decoder", type=str, default="gat")
    parser.add_argument("--loss_fn", type=str, default="sce")  # sce or mse
    parser.add_argument("--alpha_l", type=float, default=2)  # gamma in recon loss, gamma in latent loss is set to be 1
    parser.add_argument("--optimizer", type=str, default="adam")
    
    parser.add_argument("--linear_prob", action="store_true", default=False)

    
    parser.add_argument("--no_pretrain", action="store_true")
    # parser.add_argument("--load_model", action="store_true")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--use_cfg", action="store_true")
    parser.add_argument("--logging", default=False)
    parser.add_argument("--log_name", type=str, default="mg_date_time.log")
    parser.add_argument("--scheduler", action="store_true", default=False)

    parser.add_argument("--batch_size", type=int, default=512)  # not used in our setting with full batch training
    parser.add_argument("--sampling_method", type=str, default="saint", help="sampling method, `lc` or `saint`")

    parser.add_argument("--label_rate", type=float, default=1.0)

    parser.add_argument("--lam", type=float, default=1.0)  # mixing coeff in latent with recon balancing
    # parser.add_argument("--full_graph_forward", action="store_true", default=False)
    parser.add_argument("--delayed_ema_epoch", type=int, default=0)
    parser.add_argument("--replace_rate", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.996)

    parser.add_argument("--load_model", default=False)

    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()
    return args
        
    
def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name == "selu":
        return nn.SELU()
    elif name == "elu":
        return nn.ELU()
    elif name == "silu":
        return nn.SiLU()
    elif name is None:
        return nn.Identity()
    else:
        raise NotImplementedError(f"{name} is not implemented.")


def identity_norm(x):
    def func(x):
        return x
    return func

def create_norm(name):
    if name == "layernorm":
        return nn.LayerNorm
    elif name == "batchnorm":
        return nn.BatchNorm1d
    elif name == "identity":
        return identity_norm
    else:
        # print("Identity norm")
        return None


def create_optimizer(opt, model, lr, weight_decay, get_num_layer=None, get_layer_scale=None):
    opt_lower = opt.lower()
    parameters = model.parameters()
    opt_args = dict(lr=lr, weight_decay=weight_decay)

    opt_split = opt_lower.split("_")
    opt_lower = opt_split[-1]

    if opt_lower == "adam":
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == "adadelta":
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == "sgd":
        opt_args["momentum"] = 0.9
        return optim.SGD(parameters, **opt_args)
    else:
        raise NotImplementedError("Invalid optimizer")

    return optimizer


def show_occupied_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2


# -------------------
def mask_edge(graph, mask_prob):
    E = graph.num_edges()

    mask_rates = torch.ones(E) * mask_prob
    masks = torch.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)
    return mask_idx


def drop_edge(graph, drop_rate, return_edges=False):
    if drop_rate <= 0:
        return graph

    graph = graph.remove_self_loop()

    n_node = graph.num_nodes()
    edge_mask = mask_edge(graph, drop_rate)
    src, dst = graph.edges()

    nsrc = src[edge_mask]
    ndst = dst[edge_mask]

    ng = dgl.graph((nsrc, ndst), num_nodes=n_node)
    ng = ng.add_self_loop()

    return ng


def visualize(x, y, method="tsne"):
    if torch.is_tensor(x):
        x = x.cpu().numpy()
        
    if torch.is_tensor(y):
        y = y.cpu().numpy()
    
    if method == "tsne":
        func = TSNE(n_components=2)
    else:
        func = PCA(n_components=2)
    out = func.fit_transform(x)
    plt.scatter(out[:, 0], out[:, 1], c=y)
    plt.savefig("vis.png")
    

def load_best_configs(args):
    dataset_name = args.dataset
    config_path = os.path.join("configs", f"{dataset_name}.yaml")
    with open(config_path, "r") as f:
        configs = yaml.load(f, yaml.FullLoader)

    for k, v in configs.items():
        if "lr" in k or "weight_decay" in k:
            v = float(v)
        setattr(args, k, v)
    logging.info(f"----- Using best configs from {config_path} -----")

    return args



def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    scheduler = np.concatenate((warmup_schedule, schedule))
    assert len(scheduler) == epochs * niter_per_ep
    return scheduler

    

# ------ logging ------

# class TBLogger(object):
#     def __init__(self, log_path="./logging_data", name="run"):
#         super(TBLogger, self).__init__()

#         if not os.path.exists(log_path):
#             os.makedirs(log_path, exist_ok=True)

#         self.last_step = 0
#         self.log_path = log_path
#         raw_name = os.path.join(log_path, name)
#         name = raw_name
#         for i in range(1000):
#             name = raw_name + str(f"_{i}")
#             if not os.path.exists(name):
#                 break
#         self.writer = SummaryWriter(logdir=name)

#     def note(self, metrics, step=None):
#         if step is None:
#             step = self.last_step
#         for key, value in metrics.items():
#             self.writer.add_scalar(key, value, step)
#         self.last_step = step

#     def finish(self):
#         self.writer.close()


class WandbLogger(object):
    def __init__(self, log_path, project, args):
        self.log_path = log_path
        self.project = project
        self.args = args
        self.last_step = 0
        self.project = project
        self.start()

    def start(self):
        self.run = wandb.init(config=self.args, project=self.project)

    def log(self, metrics, step=None):
        if not hasattr(self, "run"):
            self.start()
        if step is None:
            step = self.last_step
        self.run.log(metrics)
        self.last_step = step

    def finish(self):
        self.run.finish()
