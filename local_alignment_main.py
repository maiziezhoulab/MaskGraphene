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
from utils_local_alignment import (
    build_args_ST,
    create_optimizer,
    set_random_seed,
    get_current_lr
)
from datasets.st_loading_utils import mclust_R
from datasets.st_loading_utils import load_DLPFC, create_dictionary_mnn, load_mHypothalamus, load_embryo, load_mMAMP
from datasets.data_proc import Cal_Spatial_Net
from models import build_model_ST


def local_alignment_loader(section_ids=["151507", "151508"], dataname="DLPFC", hvgs=5000, st_data_dir="./", sim_dir = "./"):
    print("name:", dataname)
    # hard links is a mapping matrix (2d numpy array with the size of #slice1 spot by #slice2 spot)
    if dataname == "DLPFC":
        Batch_list = []
        adj_list = []
        for section_id in section_ids:
            ad_ = load_DLPFC(root_dir=st_data_dir, section_id=section_id)
            ad_.var_names_make_unique(join="++")

            # make spot name unique
            ad_.obs_names = [x+'_'+section_id for x in ad_.obs_names]
            
            # Constructing the spatial network
            Cal_Spatial_Net(ad_, rad_cutoff=150) # the spatial network are saved in adata.uns[‘adj’]
            
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
    elif dataname == "mHypothalamus":
        Batch_list = []
        adj_list = []
        for section_id in section_ids:
            ad_ = load_mHypothalamus(root_dir=st_data_dir, section_id=section_id)
            ad_.var_names_make_unique(join="++")
        
            # make spot name unique
            ad_.obs_names = [x+'_'+section_id for x in ad_.obs_names]
            
            # Constructing the spatial network
            Cal_Spatial_Net(ad_, rad_cutoff=35) # the spatial network are saved in adata.uns[‘adj’]
            
            # Normalization
            sc.pp.normalize_total(ad_, target_sum=1e4)
            sc.pp.log1p(ad_)

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
        graph.ndata["feat"] = torch.tensor(adata_concat.X).float()
        num_features = graph.ndata["feat"].shape[1]

    ## fei added on 11/08/23
    elif dataname == "embryo":
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
    
    # fei added on 11/28/23
    elif dataname == "DLPFC_sim":
        Batch_list = []
        adj_list = []
        ad_ = load_DLPFC(root_dir=st_data_dir, section_id=section_ids[0])
        del ad_.uns['spatial']
        ad_.var_names_make_unique(join="++")
        section_id = section_ids[0]
        # make spot name unique
        ad_.obs_names = [x+'_'+section_id for x in ad_.obs_names]
        
        # Constructing the spatial network
        Cal_Spatial_Net(ad_, rad_cutoff=150) # the spatial network are saved in adata.uns[‘adj’]
        
        # Normalization
        sc.pp.highly_variable_genes(ad_, flavor="seurat_v3", n_top_genes=hvgs)
        sc.pp.normalize_total(ad_, target_sum=1e4)
        sc.pp.log1p(ad_)
        ad_ = ad_[:, ad_.var['highly_variable']]

        adj_list.append(ad_.uns['adj'])
        Batch_list.append(ad_)
        print(ad_)
        ad_ = sc.read_h5ad(osp.join(sim_dir, section_ids[1]+'.h5'))
        del ad_.obs['SpotIndex']
        del ad_.uns['spatial']
        ad_.X = ad_.X.astype(np.float32)

        ad_.obs['original_clusters'] = ad_.obs['original_clusters'].astype(object)
        ad_.var['feature_types'] = ad_.var['feature_types'].astype(object)
        ad_.var['genome'] = ad_.var['genome'].astype(object)
        ad_.var_names_make_unique(join="++")
        section_id = section_ids[1]
        # make spot name unique
        ad_.obs_names = [x+'_'+section_id for x in ad_.obs_names]
        
        # Constructing the spatial network
        Cal_Spatial_Net(ad_, rad_cutoff=150) # the spatial network are saved in adata.uns[‘adj’]
        
        # Normalization
        sc.pp.highly_variable_genes(ad_, flavor="seurat_v3", n_top_genes=hvgs)
        sc.pp.normalize_total(ad_, target_sum=1e4)
        sc.pp.log1p(ad_)
        ad_ = ad_[:, ad_.var['highly_variable']]

        adj_list.append(ad_.uns['adj'])
        Batch_list.append(ad_)
        print(ad_)
  
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
    elif dataname == "MAMP":
        Batch_list = []
        adj_list = []
        for section_id in section_ids:
            ad_ = load_mMAMP(root_dir=st_data_dir, section_id=section_id)
            ad_.var_names_make_unique(join="++")

            # make spot name unique
            ad_.obs_names = [x+'_'+section_id for x in ad_.obs_names]
            
            # Constructing the spatial network
            Cal_Spatial_Net(ad_, rad_cutoff=150) # the spatial network are saved in adata.uns[‘adj’]
            
            # Normalization
            sc.pp.highly_variable_genes(ad_, flavor="seurat_v3", n_top_genes=hvgs)
            sc.pp.normalize_total(ad_, target_sum=1e4)
            sc.pp.log1p(ad_)
            ad_ = ad_[:, ad_.var['highly_variable']]

            adj_list.append(ad_.uns['adj'])
            Batch_list.append(ad_)

        adata_concat = ad.concat(Batch_list, label="slice_name", keys=section_ids, uns_merge="same")
        #adata_concat.obs['original_clusters'] = adata_concat.obs['original_clusters'].astype('category')
        adata_concat.obs["batch_name"] = adata_concat.obs["slice_name"].astype('category')

        adj_concat = np.asarray(adj_list[0].todense())
        for batch_id in range(1,len(section_ids)):
            adj_concat = scipy.linalg.block_diag(adj_concat, np.asarray(adj_list[batch_id].todense()))

        edgeList = np.nonzero(adj_concat)
        graph = dgl.graph((edgeList[0], edgeList[1]))
        graph.ndata["feat"] = torch.tensor(adata_concat.X.todense())
        num_features = graph.ndata["feat"].shape[1]
    else:
        raise NotImplementedError
    return graph, num_features, adata_concat


def run_local_alignment(graph, model, device, ad_concat, section_ids, max_epoch, max_epoch_triplet, optimizer, scheduler, logger, num_class, use_mnn=False):
    x = graph.ndata["feat"]
    model.to(device)
    graph = graph.to(device)
    x = x.to(device)

    print("num of cluster:", num_class)
    
    """training"""
    target_nodes = torch.arange(x.shape[0], device=x.device, dtype=torch.long)
    epoch_iter = tqdm(range(max_epoch))

    # print("training local clusters ... ")
    for epoch in epoch_iter:
        model.train()
        #print(x.dtype, type(graph), target_nodes.dtype)
        loss = model(graph, x, targets=target_nodes)

        loss_dict = {"loss": loss.item()}
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        epoch_iter.set_description(f"# Epoch {epoch}: train_loss: {loss.item():.4f}")
        if logger is not None:
            loss_dict["lr"] = get_current_lr(optimizer)
            logger.log(loss_dict, step=epoch)
    
    with torch.no_grad():
        embedding = model.embed(graph, x)
    ad_concat.obsm["maskgraphene"] = embedding.cpu().detach().numpy()
    
    if use_mnn:
        epoch_iter = tqdm(range(max_epoch_triplet))
        for epoch in epoch_iter:
            if epoch % 100 == 0:
                mnn_dict = create_dictionary_mnn(ad_concat, use_rep="maskgraphene", batch_name='batch_name', k=50, iter_comb=None)
                anchor_ind = []
                positive_ind = []
                negative_ind = []
                for batch_pair in mnn_dict.keys():  # pairwise compare for multiple batches
                    batchname_list = ad_concat.obs['batch_name'][mnn_dict[batch_pair].keys()]

                    cellname_by_batch_dict = dict()
                    for batch_id in range(len(section_ids)):
                        cellname_by_batch_dict[section_ids[batch_id]] = ad_concat.obs_names[
                            ad_concat.obs['batch_name'] == section_ids[batch_id]].values

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

                    batch_as_dict = dict(zip(list(ad_concat.obs_names), range(0, ad_concat.shape[0])))
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

            if scheduler is not None:
                scheduler.step()
            epoch_iter.set_description(f"# Epoch {epoch}: train_loss: {loss.item():.4f}")
            

        with torch.no_grad():
            embedding = model.embed(graph, x)
        ad_concat.obsm["maskgraphene_mnn"] = embedding.cpu().detach().numpy()

    if use_mnn:
        mclust_R(ad_concat, modelNames='EEE', num_cluster=num_class, used_obsm='maskgraphene_mnn')
    else:
        mclust_R(ad_concat, modelNames='EEE', num_cluster=num_class, used_obsm='maskgraphene')

    Batch_list = []
    if "MA" in section_ids:
        ad__ = ad_concat[ad_concat.obs['batch_name'] == section_ids[0]]
        ma = sc.read_visium(path = '/maiziezhou_lab/Datasets/ST_datasets/mMAMP/MA', count_file='MA_filtered_feature_bc_matrix.h5')
        ma.var_names_make_unique()
        gt_dir = os.path.join('/maiziezhou_lab/Datasets/ST_datasets/mMAMP', 'MA', 'gt')
        gt_df = pd.read_csv(os.path.join(gt_dir, 'tissue_positions_list_GTs.txt'), sep='\t', header=0, index_col=0)
        ma.obs = gt_df
        ma.obs['Ground Truth'] = ma.obs['ground_truth'].astype('category')
        ma.obs_names = [x+'_'+'MA' for x in ma.obs_names]
        ad__.obs['original_clusters'] = 'unknown'
        for spot in ad__.obs_names:
            if spot in ma.obs_names:
                # Transfer the value from ma.obs['Ground Truth'] to Batch_list[0].obs['Ground Truth']
                ad__.obs.loc[spot, 'original_clusters'] = ma.obs.loc[spot, 'Ground Truth']
            else:
                # If the spot is only in Batch_list[0], set 'Ground Truth' to 'unknown'
                ad__.obs.loc[spot, 'original_clusters'] = 'unknown'
        ad__ = ad__[ad__.obs['original_clusters']!='unknown'] 
        # print("num of spots in final MA", ad__.shape[0])
        # print('mclust, ARI = %01.3f' % ari_score(ad__.obs['original_clusters'], ad__.obs['mclust']))   
        Batch_list.append(ad__)
        ad__ = ad_concat[ad_concat.obs['batch_name'] == section_ids[1]]
        Batch_list.append(ad__)

    else:
        ad_temp = ad_concat[ad_concat.obs['original_clusters']!='unknown']

        for section_id in section_ids:
            ad__ = ad_temp[ad_temp.obs['batch_name'] == section_id]
            Batch_list.append(ad__)
            print(section_id)
            print('mclust, ARI = %01.3f' % ari_score(ad__.obs['original_clusters'], ad__.obs['mclust']))
            # print("using mclust")
    
    return Batch_list, ad_concat


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

    encoder_type = args.encoder
    decoder_type = args.decoder
    replace_rate = args.replace_rate
    is_consecutive = args.consecutive_prior

    optim_type = args.optimizer 
    loss_fn = args.loss_fn

    lr = args.lr
    weight_decay = args.weight_decay
    logger = None
    use_scheduler = args.scheduler

    exp_fig_dir = args.exp_fig_dir
    st_data_dir = args.st_data_dir

    """ST loading"""
    section_ids = args.section_ids.lstrip().split(",")
    
    if not os.path.exists(os.path.join(exp_fig_dir, dataset_name+'_'.join(section_ids))):
        os.makedirs(os.path.join(exp_fig_dir, dataset_name+'_'.join(section_ids)))
    exp_fig_dir = os.path.join(exp_fig_dir, dataset_name+'_'.join(section_ids))

    for i, seed in enumerate(seeds):
        # print(f"####### Run {i} for seed {seed}")
        set_random_seed(seed)
        
        graph, num_features, ad_concat = local_alignment_loader(section_ids=section_ids, hvgs=args.hvgs, st_data_dir=st_data_dir, sim_dir=sim_dir, dataname=dataset_name, hard_links=None)
        args.num_features = num_features
        # print(args)
        model_local_ot = build_model_ST(args)
        # print(model_local_ot)
        model_local_ot.to(device)
        optimizer = create_optimizer(optim_type, model_local_ot, lr, weight_decay)

        if use_scheduler:
            logging.info("Use scheduler")
            scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epoch) ) * 0.5
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
        else:
            scheduler = None
        
        batchlist_, ad_concaT = run_local_alignment(graph, model_local_ot, device, ad_concat, section_ids, max_epoch=max_epoch, max_epoch_triplet=max_epoch_triplet, optimizer=optimizer, scheduler=scheduler, logger=logger, num_class=num_class, use_mnn=True)

        slice1 = batchlist_[0]
        slice2 = batchlist_[1]

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
    mapping_mat = scipy.sparse.csr_matrix(global_PI)
    file = open(os.path.join(exp_fig_dir, file_name+"_HL.pickle"),'wb')
    pickle.dump(mapping_mat, file)


    new_slices = paste.stack_slices_pairwise(batchlist_, mapping_mat)
    for i,L in enumerate(new_slices):
        spatial_data = L.obsm['spatial']

        output_path = os.path.join(exp_fig_dir, f"coordinates_{section_ids[i]}.csv")
        pd.DataFrame(spatial_data).to_csv(output_path, index=False)
        print(f"Saved spatial data for slice {i} to {output_path}")
    # file_name = section_ids[0]+'_'+section_ids[1] +'_a'+str(alpha_value)
    # file2 = open(os.path.join(exp_fig_dir, file_name+"_Cluster_matrix.pickle"),'wb')
    # S2 = scipy.sparse.csr_matrix(cluster_matrix)
    # pickle.dump(S2, file2)

    return global_PI, batchlist_


if __name__ == "__main__":
    args = build_args_ST()
    pi, Batch_list = localMG(args)
