import logging
import anndata
import torch

import dgl
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from ogb.nodeproppred import DglNodePropPredDataset

from sklearn.neighbors import NearestNeighbors

from sklearn.preprocessing import StandardScaler
from .st_loading_utils import load_DLPFC, load_mHypothalamus, load_mMAMP, load_embryo
import scanpy as sc
import pandas as pd
import sklearn.neighbors
import numpy as np
import scipy.sparse as sp
import paste
import ot
import os
import anndata as ad
import scipy
from scipy.sparse import csr_matrix
import networkx as nx
from sklearn.metrics.pairwise import pairwise_distances
import sklearn
from scipy.spatial.distance import cdist
ST_DICT = {
    "DLPFC", "BC", "MHypo", "MB2SAP", "Embryo", "DLPFC_sim", "MB"
}

        
def Cal_Spatial_Net(adata, rad_cutoff=None, k_cutoff=None, max_neigh=50, model='Radius', verbose=True):
    """
    Construct the spatial neighbor networks.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    rad_cutoff
        radius cutoff when model='Radius'
    k_cutoff
        The number of nearest neighbors when model='KNN'
    model
        The network construction model. When model=='Radius', the spot is connected to spots whose distance is less than rad_cutoff. When model=='KNN', the spot is connected to its first k_cutoff nearest neighbors.

    Returns
    -------
    The spatial networks are saved in adata.uns['Spatial_Net']
    """

    assert (model in ['Radius', 'KNN'])
    if verbose:
        print('------Calculating spatial graph...')
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']

    nbrs = sklearn.neighbors.NearestNeighbors(
        n_neighbors=max_neigh + 1, algorithm='ball_tree').fit(coor)
    distances, indices = nbrs.kneighbors(coor)
    if model == 'KNN':
        indices = indices[:, 1:k_cutoff + 1]
        distances = distances[:, 1:k_cutoff + 1]
    if model == 'Radius':
        indices = indices[:, 1:]
        distances = distances[:, 1:]

    KNN_list = []
    for it in range(indices.shape[0]):
        KNN_list.append(pd.DataFrame(zip([it] * indices.shape[1], indices[it, :], distances[it, :])))
    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

    Spatial_Net = KNN_df.copy()
    if model == 'Radius':
        Spatial_Net = KNN_df.loc[KNN_df['Distance'] < rad_cutoff,]
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
    # self_loops = pd.DataFrame(zip(Spatial_Net['Cell1'].unique(), Spatial_Net['Cell1'].unique(),
    #                  [0] * len((Spatial_Net['Cell1'].unique())))) ###add self loops
    # self_loops.columns = ['Cell1', 'Cell2', 'Distance']
    # Spatial_Net = pd.concat([Spatial_Net, self_loops], axis=0)

    if verbose:
        print('The graph contains %d edges, %d cells.' % (Spatial_Net.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' % (Spatial_Net.shape[0] / adata.n_obs))
    adata.uns['Spatial_Net'] = Spatial_Net

    #########
    try:
        X = pd.DataFrame(adata.X.toarray()[:, ], index=adata.obs.index, columns=adata.var.index)
    except:
        X = pd.DataFrame(adata.X[:, ], index=adata.obs.index, columns=adata.var.index)
    cells = np.array(X.index)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    if 'Spatial_Net' not in adata.uns.keys():
        raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")
        
    Spatial_Net = adata.uns['Spatial_Net']
    G_df = Spatial_Net.copy()
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)
    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    G = G + sp.eye(G.shape[0])  # self-loop
    adata.uns['adj'] = G

def nearest_mapping_matrix(arr1, arr2):
    # Step 1: Compute the Euclidean distance between each pair of points
    dist_matrix = cdist(arr1, arr2, 'euclidean')
    
    # Step 2: Create a binary matrix where each row has a single 1 at the column with the minimum distance
    mapping_matrix = np.zeros_like(dist_matrix, dtype=int)
    min_indices = np.argmin(dist_matrix, axis=1)
    
    for i, min_index in enumerate(min_indices):
        mapping_matrix[i, min_index] = 1
        
    return mapping_matrix

def Cal_Spatial_Net_feature(adata, niche_x, rad_cutoff=None, k_cutoff=None, max_neigh=50, model='Radius', verbose=True):
    """
    Construct the spatial neighbor networks.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    rad_cutoff
        radius cutoff when model='Radius'
    k_cutoff
        The number of nearest neighbors when model='KNN'
    model
        The network construction model. When model=='Radius', the spot is connected to spots whose distance is less than rad_cutoff. When model=='KNN', the spot is connected to its first k_cutoff nearest neighbors.

    Returns
    -------
    The spatial networks are saved in adata.uns['Spatial_Net']
    """

    assert (model in ['Radius', 'KNN'])
    if verbose:
        print('------Calculating spatial graph...')
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']

    nbrs = sklearn.neighbors.NearestNeighbors(
        n_neighbors=max_neigh + 1, algorithm='ball_tree').fit(coor)
    distances, indices = nbrs.kneighbors(coor)
    if model == 'KNN':
        indices = indices[:, 1:k_cutoff + 1]
        distances = distances[:, 1:k_cutoff + 1]
    if model == 'Radius':
        indices = indices[:, 1:]
        distances = distances[:, 1:]

    KNN_list = []
    for it in range(indices.shape[0]):
        KNN_list.append(pd.DataFrame(zip([it] * indices.shape[1], indices[it, :], distances[it, :])))
    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

    Spatial_Net = KNN_df.copy()
    if model == 'Radius':
        Spatial_Net = KNN_df.loc[KNN_df['Distance'] < rad_cutoff,]
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
    # self_loops = pd.DataFrame(zip(Spatial_Net['Cell1'].unique(), Spatial_Net['Cell1'].unique(),
    #                  [0] * len((Spatial_Net['Cell1'].unique())))) ###add self loops
    # self_loops.columns = ['Cell1', 'Cell2', 'Distance']
    # Spatial_Net = pd.concat([Spatial_Net, self_loops], axis=0)

    # if verbose:
    #     print('The graph contains %d edges, %d cells.' % (Spatial_Net.shape[0], adata.n_obs))
    #     print('%.4f neighbors per cell on average.' % (Spatial_Net.shape[0] / adata.n_obs))
    adata.uns['Spatial_Net'] = Spatial_Net

    #########
    try:
        X = pd.DataFrame(adata.X.toarray()[:, ], index=adata.obs.index, columns=adata.var.index)
    except:
        X = pd.DataFrame(adata.X[:, ], index=adata.obs.index, columns=adata.var.index)
    cells = np.array(X.index)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    if 'Spatial_Net' not in adata.uns.keys():
        raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")
        
    Spatial_Net = adata.uns['Spatial_Net']

    G_df = Spatial_Net.copy()
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)

    # G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    # G = get_adjacency_matrix(niche_x, k=6, metric='euclidean')
    """
    niche_data = "/home/yunfei/multimodal/UnitedNet_NatComm_data/DLPFC/adata_mrna_niche_all.h5ad"

    ad_niche = sc.read_h5ad(niche_data)
    """
    print("incorporate niche data ...")
    adj = get_adjacency_matrix(niche_x, k=3, metric='euclidean')
    adj2 = get_adjacency_matrix(adata.obsm['spatial'], k=6, metric='euclidean')

    G = non_zero_intersection(adj, adj2)

    G = G + sp.eye(G.shape[0])  # self-loop
    adata.uns['adj'] = G
    # print(np.max(G))
    # print(np.count_nonzero(G)/adata.n_obs)


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


from sklearn.neighbors import NearestNeighbors

def get_adjacency_matrix_effi(A, k=6, metric='euclidean'):
    # Use NearestNeighbors to find the k nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto', metric=metric).fit(A)  # k+1 because the first neighbor is the point itself
    distances, indices = nbrs.kneighbors(A)

    # Initialize adjacency matrix
    n_samples = A.shape[0]
    adjacency_matrix = np.zeros((n_samples, n_samples))

    # Fill the adjacency matrix
    for i in range(n_samples):
        adjacency_matrix[i, indices[i][1:]] = 1  # Skip the first index since it's the point itself

    return adjacency_matrix


def get_adj_mat_pi(ad_list, mapping_mat, k):
    """
    Get the symmetric adjacency matrix with k-nearest neighbors for entries from two AnnData slices.

    Parameters:
    - ad1: AnnData object with 4000 entries
    - ad2: AnnData object with 5000 entries
    - mapping_mat: A 4000x5000 matrix denoting the mapping relationship of entries from the two AnnData objects
    - k: The number of nearest neighbors to return

    Returns:
    - adj_mat: A symmetric adjacency matrix of shape (9000, 9000)
    """
    ad1, ad2 = ad_list[0], ad_list[1]
    n1, n2 = ad1.shape[0], ad2.shape[0]
    total_entries = n1 + n2
    
    # Ensure k is not greater than the number of combined entries
    if k > total_entries:
        raise ValueError("k cannot be greater than the total number of entries in ad1 and ad2")
    
    # Get spatial coordinates
    spatial_coords1 = ad1.obsm['spatial']
    spatial_coords2 = ad2.obsm['spatial']
    
    # Initialize adjacency matrix
    adj_mat = np.zeros((total_entries, total_entries), dtype=bool)
    
    # Find k-nearest neighbors within ad1
    nbrs1 = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(spatial_coords1)
    distances1, indices1 = nbrs1.kneighbors(spatial_coords1)
    
    # Fill in the adjacency matrix for ad1 and make it symmetric
    for i in range(n1):
        for j in indices1[i]:
            adj_mat[i, j] = True
            adj_mat[j, i] = True
    
    # Find k-nearest neighbors within ad2
    nbrs2 = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(spatial_coords2)
    distances2, indices2 = nbrs2.kneighbors(spatial_coords2)
    
    # Fill in the adjacency matrix for ad2 and make it symmetric
    for i in range(n2):
        for j in indices2[i]:
            adj_mat[n1 + i, n1 + j] = True
            adj_mat[n1 + j, n1 + i] = True
    
    
    # For each entry in ad2, find its k-nearest neighbors and use these to determine the k-nearest neighbors for the mapped entry in ad1
    for i in range(n2):
        # Find the corresponding entry in ad1 using argmax on the mapping matrix
        mapped_index = np.argmax(mapping_mat[:, i])
        
        adj_mat[n1+i, 0:n1] = adj_mat[mapped_index, 0:n1]
    
    # For each entry in ad1, find its k-nearest neighbors and use these to determine the k-nearest neighbors for the mapped entry in ad2
    for i in range(n1):
        # Find the corresponding entry in ad2 using argmax on the mapping matrix (transposed)
        mapped_index = np.argmax(mapping_mat[i, :])
        
        # for j in range(n2):
        adj_mat[i, n1:n1+n2] = adj_mat[n1+mapped_index, n1:n1+n2]
    
    return adj_mat


def non_zero_intersection(array1, array2):
    # Perform element-wise logical AND operation to identify the non-zero intersection
    intersection = (array1 != 0) & (array2 != 0)
    
    # Extract the non-zero intersection from the original arrays
    result = np.where(intersection, 1, 0)
    
    return result


def dlpfc_loader(section_ids, args):
    Batch_list_new = []
    cls_ = 0
    for section_id in section_ids:
        ad_ = load_DLPFC(root_dir=args.st_data_dir, section_id=section_id)
        ad_.var_names_make_unique(join="++")

        
        align_coord = pd.read_csv(os.path.join(args.hl_dir, 'refined_coordinates_{}.csv'.format(section_id)), index_col=0)
        ad_.obsm['spatial'] = align_coord.values
        cls_ = max(cls_, len(ad_.obs['original_clusters'].unique()))

        ad_.obs_names = [x + '_' + section_id for x in ad_.obs_names]
        Batch_list_new.append(ad_)
        # print(align_coord.values)
    
    adata_concat = anndata.concat(Batch_list_new, label="slice_name", keys=section_ids, uns_merge="same")
    adata_concat.obs['original_clusters'] = adata_concat.obs['original_clusters'].astype('category')
    adata_concat.obs["batch_name"] = adata_concat.obs["slice_name"].astype('category')

    adj_spatial = get_adjacency_matrix(adata_concat.obsm['spatial'], k=10, metric='euclidean')

    adata_concat.X = fill_missing_features(adata_concat.X.toarray(), adj_spatial)

    sc.pp.highly_variable_genes(adata_concat, flavor="seurat_v3", n_top_genes=args.hvgs)
    sc.pp.normalize_total(adata_concat, target_sum=1e4)
    sc.pp.log1p(adata_concat)
    adata_concat = adata_concat[:, adata_concat.var['highly_variable']]

    return adj_spatial, adata_concat, cls_


def dlpfc_multi_loader(section_ids, args):
    Batch_list_new = []
    cls_ = 0
    for section_id in section_ids:
        ad_ = load_DLPFC(root_dir=args.st_data_dir, section_id=section_id)
        ad_.var_names_make_unique(join="++")
        align_coord = pd.read_csv(os.path.join(args.hl_dir, 'refined_coordinates_{}.csv'.format(section_id)), index_col=0)
        ad_.obsm['spatial'] = align_coord.values
        cls_ = max(cls_, len(ad_.obs['original_clusters'].unique()))
        ad_.obs_names = [x + '_' + section_id for x in ad_.obs_names]
        Batch_list_new.append(ad_)

    adata_concat = anndata.concat(Batch_list_new, label="slice_name", keys=section_ids, uns_merge="same")
    adata_concat.obs['original_clusters'] = adata_concat.obs['original_clusters'].astype('category')
    adata_concat.obs["batch_name"] = adata_concat.obs["slice_name"].astype('category')

    adj_spatial = get_adjacency_matrix(adata_concat.obsm['spatial'], k=15, metric='euclidean')

    adata_concat.X = fill_missing_features(adata_concat.X.toarray(), adj_spatial)

    sc.pp.highly_variable_genes(adata_concat, flavor="seurat_v3", n_top_genes=args.hvgs)
    sc.pp.normalize_total(adata_concat, target_sum=1e4)
    sc.pp.log1p(adata_concat)
    adata_concat = adata_concat[:, adata_concat.var['highly_variable']]

    return adj_spatial, adata_concat, cls_


def mhypo_loader(section_ids, args):
    Batch_list = []
    cls_ = 0
    adj_list = []
    for section_id in section_ids:
        ad_ = load_mHypothalamus(root_dir=args.st_data_dir, section_id=section_id)
        ad_.var_names_make_unique(join="++")

        
        align_coord = pd.read_csv(os.path.join(args.hl_dir, 'refined_coordinates_{}.csv'.format(section_id)), index_col=0)
        ad_.obsm['spatial'] = align_coord.values
        cls_ = max(cls_, len(ad_.obs['original_clusters'].unique()))
        Cal_Spatial_Net(ad_, rad_cutoff=35) # the spatial network are saved in adata.uns[‘adj’]
        ad_.obs_names = [x + '_' + section_id for x in ad_.obs_names]
        Batch_list.append(ad_)
        adj_list.append(ad_.uns['adj'])
    
    adata_concat = ad.concat(Batch_list, label="slice_name", keys=section_ids, uns_merge="same")
    adata_concat.obs['original_clusters'] = adata_concat.obs['original_clusters'].astype('category')
    adata_concat.obs["batch_name"] = adata_concat.obs["slice_name"].astype('category')

    # adj_spatial = get_adjacency_matrix(adata_concat.obsm['spatial'], k=6, metric='euclidean')

    # adata_concat.X = fill_missing_features(adata_concat.X, adj_spatial)
    # print(non_zero_rate(adata_concat.X))

    # sc.pp.highly_variable_genes(adata_concat, flavor="seurat_v3", n_top_genes=5000)
    sc.pp.normalize_total(adata_concat, target_sum=1e4)
    sc.pp.log1p(adata_concat)
    # adata_concat = adata_concat[:, adata_concat.var['highly_variable']]

    adj_spatial = np.asarray(adj_list[0].todense())
    for batch_id in range(1,len(section_ids)):
        adj_spatial = scipy.linalg.block_diag(adj_spatial, np.asarray(adj_list[batch_id].todense()))

    # # TODO embed pi in the reader

    # assert adj_concat.shape[0] == pi.shape[0] + pi.shape[1], "adj matrix shape is not consistent with the pi matrix"

    # """keep max"""
    # # max_values = np.max(pi, axis=1)

    # # # Create a new array with zero
    # # pi_keep_argmax = np.zeros_like(pi)

    # # # Loop through each row and set the maximum value to 1 (or any other desired value)
    # # for i in range(pi.shape[0]):
    # #     pi_keep_argmax[i, np.argmax(pi[i])] = max_values[i]
    pi = nearest_mapping_matrix(Batch_list[0].obsm['spatial'], Batch_list[1].obsm['spatial'])
    # # pi = pi_keep_argmax
    # """"""

    for i in range(pi.shape[0]):
        for j in range(pi.shape[1]):
            if pi[i][j] > 0:
                adj_spatial[i][j+pi.shape[0]] = 1
                adj_spatial[j+pi.shape[0]][i] = 1

    return adj_spatial, adata_concat, cls_

def mhypo_multi_loader(section_ids, args):
    Batch_list_new = []

    cls_ = 0
    for section_id in section_ids:
        # print(args.st_data_dir)
        # print(section_id)
        ad_ = load_mHypothalamus(root_dir=args.st_data_dir, section_id=section_id)
        ad_.var_names_make_unique(join="++")

        
        align_coord = pd.read_csv(os.path.join(args.hl_dir, 'refined_coordinates_{}.csv'.format(section_id)), index_col=0)
        ad_.obsm['spatial'] = align_coord.values
        cls_ = max(cls_, len(ad_.obs['original_clusters'].unique()))

        ad_.obs_names = [x + '_' + section_id for x in ad_.obs_names]
        Batch_list_new.append(ad_)

    adata_concat = anndata.concat(Batch_list_new, label="slice_name", keys=section_ids, uns_merge="same")
    adata_concat.obs['original_clusters'] = adata_concat.obs['original_clusters'].astype('category')
    adata_concat.obs["batch_name"] = adata_concat.obs["slice_name"].astype('category')

    adj_spatial = get_adjacency_matrix(adata_concat.obsm['spatial'], k=15, metric='euclidean')

    sc.pp.normalize_total(adata_concat, target_sum=1e4)
    sc.pp.log1p(adata_concat)

    return adj_spatial, adata_concat, cls_


def MB_loader(section_ids, args):
    Batch_list = []
    adj_list = []
    cls_ = 0
    for section_id in section_ids:
        ad_ = sc.read_h5ad(os.path.join(args.st_data_dir, 'merfish_mouse_brain_slice' + str(section_id) + '.h5ad'))
        ad_.var_names_make_unique(join="++")
    
        # make spot name unique
        ad_.obs_names = [x+'_'+section_id for x in ad_.obs_names]
        # print(ad_.obs)
        # print(ad_.obs.columns)
        cls_ = max(cls_, len(ad_.obs['spa_cluster'].unique()))
        ad_.obs['original_clusters'] = ad_.obs['spa_cluster']
        align_coord = pd.read_csv(os.path.join(args.hl_dir, 'refined_coordinates_{}.csv'.format(section_id)), index_col=0)
        ad_.obsm['spatial_aligned'] = align_coord.values

        # Normalization
        sc.pp.normalize_total(ad_, target_sum=1e4)
        sc.pp.log1p(ad_)

        Batch_list.append(ad_)
    
    adata_concat = ad.concat(Batch_list, label="slice_name", keys=section_ids, uns_merge="same")
    adata_concat.obs["batch_name"] = adata_concat.obs["slice_name"].astype('category')

    adj_spatial = get_adjacency_matrix(adata_concat.obsm['spatial_aligned'], k=10, metric='euclidean')
    
    return adj_spatial, adata_concat, cls_



def ma_loader(section_ids, args):
    Batch_list = []
    adj_list = []
    cls_ = 0
    for section_id in section_ids:
        ad_ = load_mMAMP(root_dir=args.st_data_dir, section_id=section_id)
        # print(ad_)
        ad_.X = ad_.X.toarray()
        ad_.var_names_make_unique(join="++")
    
        # make spot name unique
        ad_.obs_names = [x+'_'+section_id for x in ad_.obs_names]
        
        # Constructing the spatial network
        Cal_Spatial_Net(ad_, rad_cutoff=150) # the spatial network are saved in adata.uns[‘adj’]
        
        # Normalization
        # ground_truth
        try:
            cls_ = max(cls_, len(ad_.obs['original_clusters'].unique()))
        except:
            cls_ = cls_
        
        adj_list.append(ad_.uns['adj'])
        Batch_list.append(ad_)

    with open('/maiziezhou_lab/yunfei/Projects/MaskGraphene_dev0625/aligned_coord_final/MB2SAP_mapping/S.pickle', 'rb') as f:
        mapping_mat = np.load(f, allow_pickle=True).toarray()
        mapping_mat_argmax = np.argmax(mapping_mat, axis=1)
    Batch_list[0], Batch_list[1] = simple_impute(Batch_list[0], Batch_list[1], mapping_mat_argmax)

    Batch_list_new = []
    for ad_ in Batch_list:
        sc.pp.highly_variable_genes(ad_, flavor="seurat_v3", n_top_genes=args.hvgs)
        sc.pp.normalize_total(ad_, target_sum=1e4)
        sc.pp.log1p(ad_)
        ad_ = ad_[:, ad_.var['highly_variable']]
        Batch_list_new.append(ad_)
    
    adata_concat = ad.concat(Batch_list_new, label="slice_name", keys=section_ids, uns_merge="same")
    adata_concat.obs['original_clusters'] = adata_concat.obs['original_clusters'].astype('category')
    adata_concat.obs["batch_name"] = adata_concat.obs["slice_name"].astype('category')

    adj_spatial = np.asarray(adj_list[0].todense())
    for batch_id in range(1,len(section_ids)):
        adj_spatial = scipy.linalg.block_diag(adj_spatial, np.asarray(adj_list[batch_id].todense()))
    
    # mapping_mat = np.argmax(mapping_mat, axis=1)
    for i in range(mapping_mat.shape[0]):
        for j in range(mapping_mat.shape[1]):
            if mapping_mat[i][j] > 0:
                adj_spatial[i][j+mapping_mat.shape[0]] = 1
                adj_spatial[j+mapping_mat.shape[0]][i] = 1
    
    
    return adj_spatial, adata_concat, cls_


def embryo_loader(section_ids, args):
    # /home/yunfei/spatial_benchmarking/benchmarking_data/Embryo
    Batch_list = []
    adj_list = []
    cls_ = 0
    print(section_ids)
    for section_id in section_ids:
        # sec = "E"+section_id+".h5ad"
        ad_ = sc.read_h5ad(os.path.join(args.st_data_dir, section_id))
        ad_.var_names_make_unique(join="++")
        # print(ad_.shape)
        # make spot name unique
        ad_.obs_names = [x+'_'+section_id for x in ad_.obs_names]
        # print(ad_.obs)
        # print(ad_.obs.columns)
        
        ad_.obs['original_clusters'] = ad_.obs['annotation']
        ad_.obs['x'] = ad_.obsm['spatial'].T[0]
        ad_.obs['y'] = ad_.obsm['spatial'].T[1]
        # print(len(ad_.obs['original_clusters'].unique()))
        # Constructing the spatial network
        # Cal_Spatial_Net(ad_, rad_cutoff=1.3) # the spatial network are saved in adata.uns[‘adj’]
        
        # Normalization
        # sc.pp.normalize_total(ad_, target_sum=1e4)
        # sc.pp.log1p(ad_)

        # sc.pp.highly_variable_genes(ad_, flavor="seurat_v3", n_top_genes=args.hvgs) #ensure enough common HVGs in the combined matrix
        # ad_ = ad_[:, ad_.var['highly_variable']]

        # adj_list.append(ad_.uns['adj'])
        Batch_list.append(ad_)
    cls_ = len(Batch_list[0].obs['annotation'].unique())
    id1_ = section_ids[0].split("_")[0][1:]
    id2_ = section_ids[1].split("_")[0][1:]
    print(id1_, id2_)
    mapping_mat = np.load(os.path.join('/maiziezhou_lab/yunfei/Projects/MaskGraphene_dev0625/aligned_coord_localOT/Embryo_mapping', id1_ + 'and' + id2_ + '_round1_alpha0.1_localOt_kl_iniPI_paste1_AlignmentPi.npy'.format(section_ids[1].split(".")[0])))
    print(mapping_mat.shape)
    # for i in range(mapping_mat.shape[0]):
    #     for j in range(mapping_mat.shape[1]):
    #         if mapping_mat[i][j] > 0:
    #             adj_spatial[i][j+mapping_mat.shape[0]] = 1
    #             adj_spatial[j+mapping_mat.shape[0]][i] = 1
    best_matches = np.argmax(mapping_mat, axis=1)
    # Replace each coordinate in spatial_coords1 with the corresponding coordinate in spatial_coords2
    # Batch_list[0].obsm['spatial_ori'] = Batch_list[0].obsm['spatial']
    mapped_coords1 = Batch_list[1].obsm['spatial'][best_matches]
    # Update the original obsm with the mapped coordinates
    Batch_list[0].obsm['spatial'] = mapped_coords1

    adata_concat = ad.concat(Batch_list, label="slice_name", keys=section_ids, uns_merge="same")
    # adata_concat.obs['original_clusters'] = adata_concat.obs['original_clusters'].astype('category')
    adata_concat.obs["batch_name"] = adata_concat.obs["slice_name"].astype('category')

    adj_spatial = get_adjacency_matrix(adata_concat.obsm['spatial'], k=9, metric='euclidean')

    adata_concat.X = fill_missing_features(adata_concat.X.toarray(), adj_spatial)
    # adata_concat.X = fill_extreme_features(adata_concat.X, adj_spatial)
    # print(non_zero_rate(adata_concat.X))

    sc.pp.highly_variable_genes(adata_concat, flavor="seurat_v3", n_top_genes=args.hvgs)
    sc.pp.normalize_total(adata_concat, target_sum=1e4)
    sc.pp.log1p(adata_concat)
    adata_concat = adata_concat[:, adata_concat.var['highly_variable']]
    
    
    return adj_spatial, adata_concat, cls_


def heart_ST_loader(section_ids, args):
    pass


def dlpfc_sim_loader(section_ids=['DLPFC_151673original_spotIndex.h5', 'DLPFC_151673_overlap=100%_pseudocount_0_spotIndex.h5'], args=None):

    Batch_list = []
    adj_list = []

    ad_ = load_DLPFC(root_dir=args.st_data_dir, section_id="151673")
    del ad_.uns['spatial']
    ad_.var_names_make_unique(join="++")
    ad_.obs_names = [x+'_'+section_ids[0] for x in ad_.obs_names]
    Cal_Spatial_Net(ad_, rad_cutoff=150) # the spatial network are saved in adata.uns[‘adj’]
    
    # Normalization
    sc.pp.highly_variable_genes(ad_, flavor="seurat_v3", n_top_genes=args.hvgs)
    sc.pp.normalize_total(ad_, target_sum=1e4)
    sc.pp.log1p(ad_)
    ad_ = ad_[:, ad_.var['highly_variable']]

    adj_list.append(ad_.uns['adj'])
    Batch_list.append(ad_)
    # print(ad_)
    ad_ = sc.read_h5ad(os.path.join("/maiziezhou_lab/yunfei/Projects/MaskGraphene_dev0625/aligned_coord_localOT/DLPFC151673_sim_data", section_ids[1]))
    del ad_.obs['SpotIndex']
    del ad_.uns['spatial']
    ad_.X = ad_.X.astype(np.float32)

    ad_.obs['original_clusters'] = ad_.obs['original_clusters'].astype(object)
    ad_.var['feature_types'] = ad_.var['feature_types'].astype(object)
    ad_.var['genome'] = ad_.var['genome'].astype(object)
    ad_.var_names_make_unique(join="++")
    ad_.obs_names = [x+'_'+section_ids[1] for x in ad_.obs_names]
    
    # Constructing the spatial network
    Cal_Spatial_Net(ad_, rad_cutoff=150) # the spatial network are saved in adata.uns[‘adj’]
    
    # Normalization
    sc.pp.highly_variable_genes(ad_, flavor="seurat_v3", n_top_genes=args.hvgs)
    sc.pp.normalize_total(ad_, target_sum=1e4)
    sc.pp.log1p(ad_)
    ad_ = ad_[:, ad_.var['highly_variable']]

    adj_list.append(ad_.uns['adj'])
    Batch_list.append(ad_)
    # print(ad_)

    adata_concat = anndata.concat(Batch_list, label="slice_name", keys=section_ids, uns_merge="same")
    adata_concat.obs['original_clusters'] = adata_concat.obs['original_clusters'].astype('category')
    adata_concat.obs["batch_name"] = adata_concat.obs["slice_name"].astype('category')

    adj_spatial = np.asarray(adj_list[0].todense())
    for batch_id in range(1,len(section_ids)):
        adj_spatial = scipy.linalg.block_diag(adj_spatial, np.asarray(adj_list[batch_id].todense()))
    
    # ### inter slice connection
    # mapping_mat = np.load(os.path.join('/maiziezhou_lab/yunfei/Projects/MaskGraphene_dev0625/aligned_coord_final/DLPFC_sim', '151673and{}_round1_alpha0.1_localOt_kl_iniPI_paste1_AlignmentPi.npy'.format(section_ids[1].split(".")[0])))
    # for i in range(mapping_mat.shape[0]):
    #     for j in range(mapping_mat.shape[1]):
    #         if mapping_mat[i][j] > 0:
    #             adj_spatial[i][j+mapping_mat.shape[0]] = 1
    #             adj_spatial[j+mapping_mat.shape[0]][i] = 1
    
    adata_concat.X = adata_concat.X.toarray()

    return adj_spatial, adata_concat, 7


def load_ST_dataset(dataset_name, section_ids=["151507", "151508"], args_=None):
    assert dataset_name in ST_DICT, f"Unknow dataset: {dataset_name}."
    name_ = '_'.join(args_.section_ids)
    
    if dataset_name == "DLPFC":
        if len(section_ids) == 2:
            adj_concat, adata_concat, num_classes = dlpfc_loader(section_ids, args_)
        elif len(section_ids) == 4:
            adj_concat, adata_concat, num_classes = dlpfc_multi_loader(section_ids, args_)
        else:
            print("invalid slice number")
            exit(-10)
        edgeList = np.nonzero(adj_concat)
        graph = dgl.graph((edgeList[0], edgeList[1]))
        print(adata_concat.X.shape)
        graph.ndata["feat"] = torch.tensor(adata_concat.X)

    elif "BC" in dataset_name:
        raise NotImplementedError
    elif "MHypo" in dataset_name:
        if len(section_ids) == 2:
            adj_concat, adata_concat, num_classes = mhypo_loader(section_ids, args_)
        elif len(section_ids) == 5:
            adj_concat, adata_concat, num_classes = mhypo_multi_loader(section_ids, args_)
        else:
            print("invalid slice number")
            exit(-10)
        
        edgeList = np.nonzero(adj_concat)
        graph = dgl.graph((edgeList[0], edgeList[1]))
        graph = dgl.add_self_loop(graph)
        graph.ndata["feat"] = torch.tensor(adata_concat.X).float()
    elif "MB2SAP" in dataset_name: 
        if len(section_ids) == 2:
            adj_concat, adata_concat, num_classes = ma_loader(section_ids, args_)
        # elif len(section_ids) > 2:
        #     raise NotImplementedError
        else:
            print("invalid slice number")
            exit(-10)
        
        edgeList = np.nonzero(adj_concat)
        graph = dgl.graph((edgeList[0], edgeList[1]))
        graph.ndata["feat"] = torch.tensor(adata_concat.X).float()
    elif "Embryo" in dataset_name:
        # /home/yunfei/spatial_benchmarking/benchmarking_data/Embryo
        if len(section_ids) == 2:
            adj_concat, adata_concat, num_classes = embryo_loader(section_ids, args_)
        elif len(section_ids) > 2:
            adj_concat, adata_concat, num_classes = embryo_multi_loader(section_ids, args_)
            num_classes = 19
        else:
            print("invalid slice number")
            exit(-10)
        
        print("num of class")
        print(num_classes)
        edgeList = np.nonzero(adj_concat)
        graph = dgl.graph((edgeList[0], edgeList[1]))
        graph = dgl.add_self_loop(graph)
        graph.ndata["feat"] = torch.tensor(adata_concat.X).float()
    elif dataset_name == "DLPFC_sim":
        # /home/yunfei/spatial_benchmarking/benchmarking_data/Embryo
        if len(section_ids) == 2:
            adj_concat, adata_concat, num_classes = dlpfc_sim_loader(section_ids, args_)
        else:
            print("invalid slice number")
            exit(-10)
        
        edgeList = np.nonzero(adj_concat)
        graph = dgl.graph((edgeList[0], edgeList[1]))
        graph.ndata["feat"] = torch.tensor(adata_concat.X).float()
    elif dataset_name == "MB":
        if len(section_ids) == 10:
            adj_concat, adata_concat, num_classes = MB_loader(section_ids, args_)
        else:
            print("invalid slice number")
            exit(-10)
        
        edgeList = np.nonzero(adj_concat)
        graph = dgl.graph((edgeList[0], edgeList[1]))
        graph.ndata["feat"] = torch.tensor(adata_concat.X).float()
    else:
        # print("not implemented ")
        raise NotImplementedError
    num_features = graph.ndata["feat"].shape[1]
    # num_classes = dataset.num_classes
    return graph, (num_features, num_classes), adata_concat


def simple_impute(ad1, ad2, pi):
    try:
        ad1.X = ad1.X.toarray()
        ad2.X = ad2.X.toarray()
    except:
        ad1.X = ad1.X
        ad2.X = ad2.X
    ad1_ = ad.AnnData(X=ad1.X, obs=ad1.obs, var=ad1.var, obsm=ad1.obsm)
    ad2_ = ad.AnnData(X=ad2.X, obs=ad2.obs, var=ad2.var, obsm=ad2.obsm)
    
    for i, row in enumerate(pi):
        # Find the target column index (argmax)
        target_col = np.argmax(row)
        
        # Create a mask for positions where arr1 is zero and arr2 is non-zero
        mask1 = (ad1.X[i] == 0) & (ad2.X[target_col] != 0)

        # Use the mask to fill up the zeros in arr1 with corresponding values from arr2
        ad1_.X[i, mask1] = ad2_.X[target_col, mask1]

        # Create a mask for positions where arr1 is zero and arr2 is non-zero
        mask2 = (ad2.X[target_col] == 0) & (ad1.X[i] != 0)

        # Use the mask to fill up the zeros in arr1 with corresponding values from arr2
        ad2_.X[target_col, mask2] = ad1_.X[i, mask2]

    return ad1_, ad2_


def fill_missing_features(node_features, adjacency_matrix):
    filled_features = np.copy(node_features)  # Create a copy of the original node features
    
    num_nodes, num_features = node_features.shape
    
    for node in range(num_nodes):
        neighbors = np.where(adjacency_matrix[node])[0]  # Find neighbors of the current node
        
        if neighbors.size > 0:
            neighbor_features = node_features[neighbors]  # Get features of neighbors
            avg_features = np.mean(neighbor_features, axis=0)  # Compute average features
            zero_indices = np.where(node_features[node] == 0)[0]  # Find zero feature indices
            
            if zero_indices.size > 0:
                filled_features[node, zero_indices] = avg_features[zero_indices]  # Fill zero features with average
            
    return filled_features


def fill_extreme_features(node_features, adjacency_matrix):
    filled_features = np.copy(node_features)  # Create a copy of the original node features
    
    num_nodes, num_features = node_features.shape
    
    for node in range(num_nodes):
        neighbors = np.where(adjacency_matrix[node])[0]  # Find neighbors of the current node
        
        if neighbors.size > 0:
            neighbor_features = node_features[neighbors]  # Get features of neighbors
            avg_features = np.mean(neighbor_features, axis=0)  # Compute average features
            
            zero_indices = np.where(node_features[node] == 0)[0]  # Find zero feature indices
            extreme_indices = np.where((node_features[node] > 1.5 * avg_features) | 
                                       (node_features[node] < 0.5 * avg_features))[0]  # Find extreme value indices
            
            if zero_indices.size > 0:
                filled_features[node, zero_indices] = avg_features[zero_indices]  # Fill zero features with average
            
            if extreme_indices.size > 0:
                filled_features[node, extreme_indices] = avg_features[extreme_indices]  # Replace extreme values with average
            
    return filled_features


# def fill_missing_features(node_features, adjacency_matrix):
#     filled_features = np.copy(node_features)  # Create a copy of the original node features
    
#     num_nodes = node_features.shape[0]
    
#     # Compute degree matrix
#     degree_matrix = np.diag(adjacency_matrix.sum(axis=1))
    
#     # Compute the inverse of the degree matrix
#     inverse_degree_matrix = np.linalg.inv(degree_matrix)
    
#     # Compute the Laplacian matrix
#     laplacian_matrix = degree_matrix - adjacency_matrix
    
#     # Compute the normalized Laplacian matrix
#     normalized_laplacian = np.dot(inverse_degree_matrix, laplacian_matrix)
    
#     # Fill missing features
#     for node in range(num_nodes):
#         zero_indices = np.where(node_features[node] == 0)[0]  # Find zero feature indices
        
#         if len(zero_indices) > 0:
#             filled_features[node, zero_indices] = np.dot(normalized_laplacian[node], node_features)  # Fill zero features with average
            
#     return filled_features


def non_zero_rate(matrix):
    total_elements = matrix.size
    non_zero_elements = np.count_nonzero(matrix)
    rate = non_zero_elements / total_elements
    return rate

def preprocess(graph):
    # make bidirected
    if "feat" in graph.ndata:
        feat = graph.ndata["feat"]
    else:
        feat = None
    src, dst = graph.all_edges()
    # graph.add_edges(dst, src)
    graph = dgl.to_bidirected(graph)
    if feat is not None:
        graph.ndata["feat"] = feat

    # add self-loop
    graph = graph.remove_self_loop().add_self_loop()
    # graph.create_formats_()
    return graph


if __name__ == '__main__':
    g, (num_feats, num_c) = load_ST_dataset(dataset_name="DLPFC", section_ids=["151507", "151508"])
