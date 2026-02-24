import logging
import numpy as np
from tqdm import tqdm
import torch
import pandas as pd
import dgl
import paste
import anndata as ad
import os.path as osp
import sys
from scipy.sparse import csr_matrix
import scipy
import ot
import os
import pickle
import scanpy as sc
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.pairwise import pairwise_distances
from utils_local_alignment import (
    build_args_ST,
    create_optimizer,
    set_random_seed,
    show_occupied_memory
)
from datasets.lc_sampler import setup_training_dataloder
from datasets.st_loading_utils import mclust_R
from datasets.st_loading_utils import create_dictionary_mnn, load_embryo
from datasets.data_proc import Cal_Spatial_Net
from models import build_model_ST


def get_adjacency_matrix(A, k=6, metric='euclidean'):
    # Calculate pairwise distances
    distances = pairwise_distances(A, metric=metric)

    # Initialize adjacency matrix
    n_samples = A.shape[0]
    adjacency_matrix = np.zeros((n_samples, n_samples))

    # Find k nearest neighbors
    for i in range(n_samples):
        indices = np.argsort(distances[i])[1:k+1]  # Exclude the sample itself
        adjacency_matrix[i, indices] = 1

    return adjacency_matrix


def local_alignment_loader(section_ids=["E16.5_E1S1.h5ad", "E15.5_E1S1.h5ad"], dataname="embryo", hvgs=5000, st_data_dir="./"):
    # E9.5 - E16.5_E1S1.h5ad
    if dataname == "embryo":
        Batch_list = []
        adj_list = []
        for section_id in section_ids:
            ad_ = load_embryo(root_dir=st_data_dir, section_id=section_id)
            ad_.var_names_make_unique(join="++")
            # ad_.X = ad_.X.toarray()
            # make spot name unique
            ad_.obs_names = [x+'_'+section_id for x in ad_.obs_names]
            
            # Constructing the spatial network
            Cal_Spatial_Net(ad_, rad_cutoff=1.5) # the spatial network are saved in adata.uns[‘adj’]
            
            # Normalization
            sc.pp.highly_variable_genes(ad_, flavor="seurat_v3", n_top_genes=hvgs)
            sc.pp.normalize_total(ad_, target_sum=1e4)
            sc.pp.log1p(ad_)
            ad_ = ad_[:, ad_.var['highly_variable']]

            adj_list.append(ad_.uns['adj'])
            Batch_list.append(ad_)
        adata_concat = ad.concat(Batch_list, label="slice_name", keys=section_ids, uns_merge="same")
        adata_concat.obs['original_clusters'] = adata_concat.obs['original_clusters'].astype('category')
        adata_concat.obs["batch_name"] = adata_concat.obs["slice_name"].astype('category')

        adj_concat = np.asarray(adj_list[0].todense())
        for batch_id in range(1,len(section_ids)):
            adj_concat = scipy.linalg.block_diag(adj_concat, np.asarray(adj_list[batch_id].todense()))

        edgeList = np.nonzero(adj_concat)
        graph = dgl.graph((edgeList[0], edgeList[1]))
        graph.ndata["feat"] = torch.tensor(adata_concat.X.todense())
        num_features = graph.ndata["feat"].shape[1]
    return graph, num_features, adata_concat    


def run_init(dataloader, model, device, ad_concat, max_epoch, optimizer, scheduler):
    logging.critical("Start training ...")

    epoch_bar = tqdm(range(max_epoch), desc="Epochs")
    for epoch in epoch_bar:
        epoch_iter = tqdm(dataloader, desc=f"Epoch {epoch+1}/{max_epoch}", leave=False)
        # assert (graph.in_degrees() > 0).all(), "after loading"
        model.train()
        optimizer.zero_grad()

        total_loss = 0
        num_instances = 0
        for batch_g in epoch_iter:
            batch_g, targets, _, node_idx = batch_g
            batch_g = batch_g.to(device)
            x = batch_g.ndata.pop("feat")
            loss = model(batch_g, x, targets=targets)

            batch_size = x.shape[0]
            # Accumulate loss (still part of the graph)
            total_loss += loss * batch_size
            num_instances += batch_size
            epoch_iter.set_description(f"train_loss: {loss.item():.4f}, Memory: {show_occupied_memory():.2f} MB")

        # Compute average loss and backpropagate
        avg_loss = total_loss / num_instances
        avg_loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        epoch_bar.set_description(f"train_loss: {avg_loss.item():.4f}, Memory: {show_occupied_memory():.2f} MB")
    
    with torch.no_grad():
        model.eval()
        all_embeddings = []

        for batch_g in tqdm(dataloader, desc="Embedding"):
            batch_g, targets, _, node_idx = batch_g
            batch_g = batch_g.to(device)
            x = batch_g.ndata.pop("feat")

            z = model.embed(batch_g, x)  # shape: [#nodes in batch, hidden_dim]
            all_embeddings.append(z.cpu())

        z_full = torch.cat(all_embeddings, dim=0)
    ad_concat.obsm["maskgraphene"] = z_full.numpy()

    return model, ad_concat


def run_init_softlinks(model, dataloader, optimizer, max_epoch, device, adata_concat_, scheduler, logger=None, key_="maskgraphene_mnn", iter_comb=None):
    logging.critical("Start training with Triplet Loss...")
    section_ids = np.array(adata_concat_.obs['batch_name'].unique())

    epoch_bar = tqdm(range(max_epoch), desc="Epochs")
    for epoch in epoch_bar:
        if epoch % 100 == 0:
            mnn_dict = create_dictionary_mnn(adata_concat_, use_rep="maskgraphene", batch_name='batch_name', k=50, iter_comb=iter_comb)
            anchor_ind = []
            positive_ind = []
            negative_ind = []
            for batch_pair in mnn_dict.keys():  # pairwise compare for multiple batches
                batchname_list = adata_concat_.obs['batch_name'][mnn_dict[batch_pair].keys()]
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
        losses = []
        num_instances = 0

        epoch_iter = tqdm(dataloader, desc=f"Epoch {epoch+1}/{max_epoch}", leave=False)
        for batch_g in epoch_iter:
            model.train()
            batch_g, targets, _, _ = batch_g
            batch_g = batch_g.to(device)
            x = batch_g.ndata.pop("feat")

            optimizer.zero_grad()
            loss = model(batch_g, x, targets=targets)  # classification loss
            # loss.backward()
            # optimizer.step()

            batch_size = x.shape[0]
            losses.append(loss * batch_size)
            num_instances += batch_size

        if scheduler is not None:
            scheduler.step()
        
        with torch.no_grad():
            all_embeddings = []
            for batch_g in dataloader:
                batch_g, targets, _, _ = batch_g
                batch_g = batch_g.to(device)
                x = batch_g.ndata.pop("feat")
                z = model.embed(batch_g, x)
                all_embeddings.append(z.cpu())
        z = torch.cat(all_embeddings, dim=0)

        anchor_arr = z[anchor_ind,]
        positive_arr = z[positive_ind,]
        negative_arr = z[negative_ind,]

        triplet_loss = torch.nn.TripletMarginLoss(margin=1, p=2, reduction='mean')
        tri_output = triplet_loss(anchor_arr, positive_arr, negative_arr)

        avg_loss = sum(losses) / num_instances
        loss = avg_loss + tri_output
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()
        # loss_dict = {"loss": loss.item()}
        epoch_bar.set_description(f"train_loss: {avg_loss.item():.4f}, Memory: {show_occupied_memory():.2f} MB")
        
    # Final embedding
    model.eval()
    with torch.no_grad():
        all_embeddings = []
        for batch_g in dataloader:
            batch_g, targets, _, _ = batch_g
            batch_g = batch_g.to(device)
            x = batch_g.ndata.pop("feat")
            z = model.embed(batch_g, x)
            all_embeddings.append(z.cpu())
        z_final = torch.cat(all_embeddings, dim=0)

    adata_concat_.obsm[key_] = z_final.numpy()
    
    return model, adata_concat_


def localMG(args):
    device = args.device if args.device >= 0 else "cpu"
    seeds = args.seeds
    dataset_name = args.dataset
    sim_dir = args.sim_dir
    alpha_value = args.alpha_value
    hard_link_dir = args.hard_link_dir
    ROUND = args.ROUND

    max_epoch = args.max_epoch
    max_epoch_triplet = args.max_epoch_triplet
    num_hidden = args.num_hidden
    #original: num_layers = args.num_layers
    num_class = args.num_class

    # encoder_type = args.encoder
    # decoder_type = args.decoder
    # replace_rate = args.replace_rate
    # is_consecutive = args.consecutive_prior

    optim_type = args.optimizer 
    loss_fn = args.loss_fn

    lr = args.lr
    weight_decay = args.weight_decay
    logger = None
    use_scheduler = args.scheduler

    exp_fig_dir = args.exp_fig_dir
    st_data_dir = args.st_data_dir

    """ST loading"""
    if args.section_ids == None:
        # section_ids = ['E15.5_E1S1', 'E16.5_E1S1']
        # section_ids = ['E11.5_E1S1', 'E12.5_E1S1']
        section_ids = ['E12.5_E1S1', 'E13.5_E1S1']
        # section_ids = ['E13.5_E1S1', 'E14.5_E1S1']
        # section_ids = ['E14.5_E1S1', 'E15.5_E1S1']
        st_data_dir = '/maiziezhou_lab/yunfei/Projects/spatial_benchmarking/benchmarking_data/Embryo'
        # args.hl_dir = '/maiziezhou_lab/yunfei/Projects/MaskGraphene_dev0625/aligned_coord_final/DLPFC'
        exp_fig_dir = '/maiziezhou_lab/yunfei/Projects/MaskGraphene_revision/embryo_multi_gpu'
        dataset_name='embryo'
        use_mnn = True
        mini_bs = 16384 # 1024
        args.num_hidden = "512,32"
        use_scheduler=True
        max_epoch=200
    else:
        section_ids = args.section_ids.lstrip().split(",")
    
    if not os.path.exists(os.path.join(exp_fig_dir, dataset_name+'_'.join(section_ids))):
        os.makedirs(os.path.join(exp_fig_dir, dataset_name+'_'.join(section_ids)))
    exp_fig_dir = os.path.join(exp_fig_dir, dataset_name+'_'.join(section_ids))

    # for i, seed in enumerate(seeds):
    # print(f"####### Run {i} for seed {seed}")
    seed = seeds[0]
    set_random_seed(seed)
    
    graph, num_features, ad_concat = local_alignment_loader(section_ids=section_ids, hvgs=args.hvgs, st_data_dir=st_data_dir, dataname=dataset_name)
    
    
    num_class = len(ad_concat.obs['original_clusters'].unique())
    args.num_features = num_features
    x = graph.ndata["feat"]
    model_local_ot = build_model_ST(args)
    print(model_local_ot)
    model_local_ot.to(device)
    optimizer = create_optimizer(optim_type, model_local_ot, lr, weight_decay)

    if use_scheduler:
        logging.info("Use scheduler")
        scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epoch) ) * 0.5
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
    else:
        scheduler = None
    
    graph_nodes = graph.num_nodes()  # or graph.shape[0] if using a tensor-like graph object

    # Fake ego_graph_nodes: each node is its own subgraph (for simplicity)
    ego_graph_nodes = [np.array([i]) for i in range(graph_nodes)]
    dataloader = setup_training_dataloder(
        'lc', ego_graph_nodes, graph, x, batch_size=mini_bs, drop_edge_rate=0)
    
    model_local_ot, ad_concat1 = run_init(dataloader, model_local_ot, device, ad_concat, max_epoch=max_epoch, optimizer=optimizer, scheduler=scheduler)
    if use_mnn:
        # (model, dataloader, optimizer, max_epoch, device, adata_concat_, scheduler, logger=None, key_="MG_triplet", iter_comb=None)
        model, ad_concat2 = run_init_softlinks(model_local_ot, dataloader, optimizer, max_epoch, device, ad_concat1, scheduler)
        mclust_R(ad_concat2, modelNames='EEE', num_cluster=num_class, used_obsm='maskgraphene_mnn')
    else:
        mclust_R(ad_concat2, modelNames='EEE', num_cluster=num_class, used_obsm='maskgraphene')

    ad_temp = ad_concat2[ad_concat2.obs['original_clusters']!='unknown']
    Batch_list = []
    for section_id in section_ids:
        ad_ = ad_temp[ad_temp.obs['batch_name'] == section_id]
        Batch_list.append(ad_)
        print(section_id)    

    slice1 = Batch_list[0]
    slice2 = Batch_list[1]

    global_PI = np.zeros((len(slice1.obs.index), len(slice2.obs.index)))
    slice1_idx_mapping = {}
    slice2_idx_mapping = {}
    for i in range(len(slice1.obs.index)):
        slice1_idx_mapping[slice1.obs.index[i]] = i
    for i in range(len(slice2.obs.index)):
        slice2_idx_mapping[slice2.obs.index[i]] = i
    
    for i in range(num_class):
        print("run for cluster:", i)
        subslice1 = slice1[slice1.obs['mclust']==i+1]
        subslice2 = slice2[slice2.obs['mclust']==i+1]
        if subslice1.shape[0]>0 and subslice2.shape[0]>0:
            if subslice1.shape[0]>1 and subslice2.shape[0]>1: 
                pi00 = paste.match_spots_using_spatial_heuristic(subslice1.obsm['spatial'], subslice2.obsm['spatial'], use_ot=True)
                local_PI = paste.pairwise_align(subslice1, subslice2, alpha=alpha_value, dissimilarity='kl', use_rep=None, norm=True, verbose=True, G_init=pi00, use_gpu = True, backend = ot.backend.TorchBackend())
            else:  # if there is only one spot in a slice, spatial dissimilarity can't be normalized
                local_PI = paste.pairwise_align(subslice1, subslice2, alpha=alpha_value, dissimilarity='kl', use_rep=None, norm=False, verbose=True, G_init=None, use_gpu = True, backend = ot.backend.TorchBackend())
            for ii in range(local_PI.shape[0]):
                for jj in range(local_PI.shape[1]):
                    global_PI[slice1_idx_mapping[subslice1.obs.index[ii]]][slice2_idx_mapping[subslice2.obs.index[jj]]] = local_PI[ii][jj]
                    # cluster_matrix[slice1_idx_mapping[subslice1.obs.index[ii]]][slice2_idx_mapping[subslice2.obs.index[jj]]] = i
        else:
            return None

    file_name = section_ids[0]+'_'+section_ids[1] +'_'+str(alpha_value)
    


    new_slices = paste.stack_slices_pairwise(Batch_list, global_PI)
    for i,L in enumerate(new_slices):
        spatial_data = L.obsm['spatial']

        output_path = os.path.join(exp_fig_dir, f"coordinates_{section_ids[i]}.csv")
        pd.DataFrame(spatial_data).to_csv(output_path, index=False)
        print(f"Saved spatial data for slice {i} to {output_path}")
    
    mapping_mat = scipy.sparse.csr_matrix(global_PI)
    file = open(os.path.join(exp_fig_dir, file_name+"_HL.pickle"),'wb')
    pickle.dump(mapping_mat, file)

    return global_PI, Batch_list


if __name__ == "__main__":
    args = build_args_ST()
    pi, Batch_list = localMG(args)
