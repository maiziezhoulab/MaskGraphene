import scanpy as sc
import os
import pandas as pd
import numpy as np
import anndata
import matplotlib.pyplot as plt
import itertools
import networkx as nx
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
import hnswlib
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score as ari_score
import os
import matplotlib.lines as mlines


plt.rcParams['figure.figsize'] = (9.0, 9.0)
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['lines.markersize'] = 10

SMALL_SIZE = 20
MEDIUM_SIZE = 30
BIGGER_SIZE = 35

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rcParams['axes.facecolor'] = 'white'


def load_DLPFC(root_dir='../benchmarking_data/DLPFC12', section_id='151507'):
    # 151507, ..., 151676 12 in total
    ad = sc.read_visium(path=os.path.join(root_dir, section_id), count_file=section_id+'_filtered_feature_bc_matrix.h5')
    ad.var_names_make_unique()

    gt_dir = os.path.join(root_dir, section_id, 'gt')
    gt_df = pd.read_csv(os.path.join(gt_dir, 'tissue_positions_list_GTs.txt'), sep=',', header=None, index_col=0)
    ad.obs['original_clusters'] = gt_df.loc[:, 6]
    keep_bcs = ad.obs.dropna().index
    ad = ad[keep_bcs].copy()
    ad.obs['original_clusters'] = ad.obs['original_clusters'].astype(int).astype(str)
    # print(ad.obs)
    return ad


# for loading mHypothalamus data
# already preprocessed? Xs are floats
def load_mHypothalamus(root_dir='/home/yunfei/spatial_benchmarking/benchmarking_data/mHypothalamus', section_id='0.26'):
    # section id = '0.26', '0.21', '0.16', '0.11', '0.06', '0.01', '-0.04', '-0.09', '-0.14', '-0.19', '-0.24', '-0.29' 12 in total
    # cluster =     15      15      14      15      15      15      14       15       15       15      16        15
    info_file = os.path.join(root_dir, 'MERFISH_Animal1_info.xlsx')
    cnts_file = os.path.join(root_dir, 'MERFISH_Animal1_cnts.xlsx')
    xls_cnts = pd.ExcelFile(cnts_file)
    # print(xls_cnts.sheet_names)
    df_cnts = pd.read_excel(xls_cnts, section_id)

    xls_info = pd.ExcelFile(info_file)
    df_info = pd.read_excel(xls_info, section_id)
    # print(df_cnts, df_info)
    spatial_X = df_info.to_numpy()
    obs_ = df_info
    if len(df_info.columns) == 5:
        obs_.columns = ['psuedo_barcodes', 'x', 'y', 'original_clusters', 'Neuron_cluster_ID']
    elif len(df_info.columns) == 6:
        obs_.columns = ['psuedo_barcodes', 'x', 'y', 'cell_types', 'Neuron_cluster_ID', 'original_clusters']
        # print(section_id)
        # print(obs_['z'].nunique())
    obs_.index = obs_['psuedo_barcodes'].tolist()
    # print(obs_)

    var_ = df_cnts.iloc[:, 0]
    var_ = pd.DataFrame(var_)
    # print(var_)

    ad = anndata.AnnData(X=df_cnts.iloc[:, 1:].T, obs=obs_, var=var_)
    spatial = np.vstack((ad.obs['x'].to_numpy(), ad.obs['y'].to_numpy()))
    ad.obsm['spatial'] = spatial.T
    return ad


# cluster = 52
def load_mMAMP(root_dir='/home/yunfei/spatial_benchmarking/benchmarking_data/mMAMP', section_id='MA'):
    fn = os.path.join(root_dir, section_id, section_id + "1.h5ad")
    ad = sc.read_h5ad(filename=fn)
    ad.var_names_make_unique()

    try:
        gt_dir = os.path.join(root_dir, section_id, 'gt')
        gt_df = pd.read_csv(os.path.join(gt_dir, 'tissue_positions_list_GTs.txt'), sep='\t', header=0, index_col=0)

        # Ensure all barcodes in `ad` are included in the ground truth DataFrame
        gt_df = gt_df.reindex(ad.obs.index)

        # Fill missing ground truth values with 'NA'
        gt_df['ground_truth'] = gt_df['ground_truth'].fillna('NA')

        # Add the ground truth column to `ad.obs`
        ad.obs['original_clusters'] = gt_df['ground_truth']
    except:
        ad.obs['original_clusters'] = 'NA'
        print("no gt available")
        
    spatial = np.vstack((ad.obs['x4'].to_numpy(), ad.obs['x5'].to_numpy()))
    ad.obsm['spatial'] = spatial.T
    return ad


def load_embryo(root_dir='/home/yunfei/spatial_benchmarking/benchmarking_data/Embryo/', section_id='E11.5_E1S1'):
    ad = sc.read_h5ad(os.path.join(root_dir, section_id + ".h5ad"))
    ad.X = ad.layers['count']
    ad.var_names_make_unique()
    # make spot name unique
    # adata.obs_names = [x + '_' + section_id for x in adata.obs_names]
    ad.obs['original_clusters'] = ad.obs['annotation']
    return ad


def mnn(ds1, ds2, names1, names2, knn=20, save_on_disk=True, approx=True):
    if approx:
        # Find nearest neighbors in first direction.
        # output KNN point for each point in ds1.  match1 is a set(): (points in names1, points in names2), the size of the set is ds1.shape[0]*knn
        match1 = nn_approx(ds1, ds2, names1, names2, knn=knn)  # , save_on_disk = save_on_disk)
        # Find nearest neighbors in second direction.
        match2 = nn_approx(ds2, ds1, names2, names1, knn=knn)  # , save_on_disk = save_on_disk)
    else:
        match1 = nn(ds1, ds2, names1, names2, knn=knn)
        match2 = nn(ds2, ds1, names2, names1, knn=knn)
    # Compute mutual nearest neighbors.
    mutual = match1 & set([(b, a) for a, b in match2])

    return mutual


def nn_approx(ds1, ds2, names1, names2, knn=50):
    dim = ds2.shape[1]
    num_elements = ds2.shape[0]
    p = hnswlib.Index(space='l2', dim=dim)
    p.init_index(max_elements=num_elements, ef_construction=100, M=16)
    p.set_ef(10)
    p.add_items(ds2)
    ind, distances = p.knn_query(ds1, k=knn)
    match = set()
    for a, b in zip(range(ds1.shape[0]), ind):
        for b_i in b:
            match.add((names1[a], names2[b_i]))
    return match


def nn(ds1, ds2, names1, names2, knn=50, metric_p=2):
    # Find nearest neighbors of first dataset.
    nn_ = NearestNeighbors(knn, p=metric_p)
    nn_.fit(ds2)
    ind = nn_.kneighbors(ds1, return_distance=False)

    match = set()
    for a, b in zip(range(ds1.shape[0]), ind):
        for b_i in b:
            match.add((names1[a], names2[b_i]))

    return match


def create_dictionary_mnn(adata, use_rep, batch_name, k=150, save_on_disk=True, approx=True, verbose=1, iter_comb=None):
    cell_names = adata.obs_names  # obs index (barcodes) of all spots

    batch_list = adata.obs[batch_name]
    datasets = []
    datasets_pcs = []
    cells = []  # a list of indexes of each slice separately
    for i in batch_list.unique():
        datasets.append(adata[batch_list == i])
        datasets_pcs.append(adata[batch_list == i].obsm[use_rep])
        cells.append(cell_names[batch_list == i])

    batch_name_df = pd.DataFrame(np.array(batch_list.unique()))
    # print(batch_name_df)
    mnns = dict()

    if iter_comb is None:  # pairs of slices
        iter_comb = list(itertools.combinations(range(len(cells)), 2))

    for comb in iter_comb:
        i = comb[0]
        j = comb[1]
        key_name1 = batch_name_df.loc[comb[0]].values[0] + "_" + batch_name_df.loc[comb[1]].values[0]
        mnns[
            key_name1] = {}  # for multiple-slice setting, the key_names1 can avoid the mnns replaced by previous slice-pair
        # if (verbose > 0):
        #     print('Processing datasets {}'.format((i, j)))

        # mapping: new -> ref (151674 -> 151673)
        new = list(cells[j])
        ref = list(cells[i])
        # print(new)
        # print(ref)

        ds1 = adata[new].obsm[use_rep]
        ds2 = adata[ref].obsm[use_rep]
        names1 = new
        names2 = ref
        # if k>1，one point in ds1 may have multiple MNN points in ds2.
        match = mnn(ds1, ds2, names1, names2, knn=k, save_on_disk=save_on_disk, approx=approx)

        G = nx.Graph()
        G.add_edges_from(match)
        node_names = np.array(G.nodes)
        anchors = list(node_names)
        adj = nx.adjacency_matrix(G)
        tmp = np.split(adj.indices, adj.indptr[1:-1])

        for i in range(0, len(anchors)):
            key = anchors[i]
            i = tmp[i]
            names = list(node_names[i])
            mnns[key_name1][key] = names
    return (mnns)


def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='MDOT', used_obs='mclust', random_seed=666):
    """
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """

    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    # print(adata.obsm[used_obsm])
    res = rmclust(adata.obsm[used_obsm], num_cluster, modelNames)

    # print(res)
    mclust_res = np.array(res[-2])

    adata.obs[used_obs] = mclust_res
    adata.obs[used_obs] = adata.obs[used_obs].astype('int')
    # adata.obs['mclust'] = adata.obs['mclust'].astype('string')
    adata.obs[used_obs] = adata.obs[used_obs].astype('category')
    return adata


# def gmm_scikit(adata, num_cluster, modelNames='gmm', used_obsm='MDOT', random_seed=666):
#     X = adata.obsm[used_obsm]
#     bgm = GaussianMixture(n_components=num_cluster, random_state=random_seed).fit(X)
#     # bgm.means_
#     lbls_gmm = bgm.predict(X)

#     adata.obs['mclust'] = lbls_gmm
#     adata.obs['mclust'] = adata.obs['mclust'].astype('int')
#     # adata.obs['mclust'] = adata.obs['mclust'].astype('string')
#     adata.obs['mclust'] = adata.obs['mclust'].astype('category')
#     return adata


# visualize everything for sanity check
def anndata_visualization(ad, fname, save_folder='/home/yunfei/spatial_benchmarking/benchmarking_data/gt_visualization',
                          col_name='original_clusters', spot_size=150):
    sc.pl.spatial(ad,
                  color=[col_name],
                  title=[col_name],
                  show=True, spot_size=spot_size)
    plt.savefig(os.path.join(save_folder, fname + ".pdf"))


def visualization_umap_spatial(ad_temp, section_ids, exp_fig_dir, dataset_name, num_iter, identifier, num_class, use_key=""):

    mclust_R(ad_temp, modelNames='EEE', num_cluster=num_class, used_obsm=use_key)

    """umap"""
    sc.pp.neighbors(ad_temp, use_rep=use_key, random_state=666)
    sc.tl.umap(ad_temp, random_state=666)

    if len(section_ids) == 2:
        section_color = ['#f8766d', '#7cae00'] # ['#f8766d', '#7cae00', '#00bfc4', '#c77cff']
    elif len(section_ids) ==4:
        section_color = ['#f8766d', '#7cae00', '#00bfc4', '#c77cff']
    elif len(section_ids) ==5:
        section_color = ['#f8766d', '#7cae00', '#00bfc4', '#c77cff', '#f8aec4']
    elif len(section_ids) ==10:
        section_color = [
        '#1f77b4',  # blue
        '#ff7f0e',  # orange
        '#2ca02c',  # green
        '#d62728',  # red
        '#9467bd',  # purple
        '#8c564b',  # brown
        '#e377c2',  # pink
        '#7f7f7f',  # gray
        '#bcbd22',  # yellow-green
        '#17becf'   # cyan
                        ]
    else:
        print("slice num not correct")
        exit(-1)
    section_color_dict = dict(zip(section_ids, section_color))
    ad_temp.uns['batch_name_colors'] = [section_color_dict[x] for x in ad_temp.obs.batch_name.cat.categories]

    # fig = sc.pl.umap(ad_temp, color=['batch_name', 'original_clusters', 'mclust'], ncols=3, wspace=0.5,
    #                  return_fig=True)
    # fig.tight_layout()  # Adjust subplots to prevent overlap
    # fig.savefig(os.path.join(exp_fig_dir, dataset_name + '_'.join(section_ids) + str(num_iter) + identifier + "_umap.pdf"))

    """visualization"""
    if dataset_name == "MHypo":
        spot_size = 25
        title_size = 12
    elif dataset_name == "DLPFC" or dataset_name == "DLPFC_sim":
        spot_size = 75
        title_size = 12
    elif dataset_name == "BC":
        spot_size = 175
        title_size = 12
    elif dataset_name == "MB2SAP":
        spot_size = 120
        title_size = 26
    elif dataset_name == "Embryo":
        spot_size = 1
        title_size = 12
    elif dataset_name == "merfishMB":
        spot_size = 25
        title_size = 12
    else:
        raise NotImplementedError

    Batch_list = []
    ARI_list = []
    i = 0
    for section_id in section_ids:
        Batch_list.append(ad_temp[ad_temp.obs['batch_name'] == section_id])
        ARI_list.append(round(ari_score(Batch_list[i].obs['original_clusters'], Batch_list[i].obs['mclust']), 3))
        i+=1
    if dataset_name == "Embryo":
        for i in range(len(Batch_list)):
            Batch_list[i].obsm['spatial'] = Batch_list[i].obs[['x', 'y']].to_numpy()

    if dataset_name != "MB2SAP":
        i = 0
        fig, ax = plt.subplots(1, 2*len(section_ids), figsize=(10*len(section_ids), 10))  # gridspec_kw={'wspace': 0.05, 'hspace': 0.1}
        for i in range(len(section_ids)):
            if dataset_name == 'merfishMB':
                try:
                    Batch_list[i].obsm['spatial'] = Batch_list[i].obsm['spatial'][['X', 'Y']].to_numpy()
                except:
                    Batch_list[i].obsm['spatial'] = Batch_list[i].obsm['spatial']
            else:
                pass
            _sc_0 = sc.pl.spatial(Batch_list[i], img_key=None, color=['mclust'], title=[''],
                                legend_fontsize=9, show=False, ax=ax[2*i], frameon=False,
                                spot_size=spot_size)
            _sc_0[0].set_title("ARI=" + str(ARI_list[i]), size=title_size)
            _sc_0_gt = sc.pl.spatial(Batch_list[i], img_key=None, color=['original_clusters'], title=[''],
                                    legend_fontsize=9, show=False, ax=ax[2*i+1], frameon=False,
                                    spot_size=spot_size)
            _sc_0_gt[0].set_title("Ground Truth", size=title_size)

        plt.tight_layout()
        plt.savefig(os.path.join(exp_fig_dir, dataset_name + '_'.join(section_ids) + str(num_iter) + identifier + "_viz.pdf"))
    else:
        colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', 
                '#7f7f7f', '#bcbd22', '#17becf', '#2ecc71', '#3498db', '#e74c3c', '#9b59b6', 
                '#34495e', '#f1c40f', '#d35400', '#c0392b', '#7d3c98', '#1abc9c', '#2980b9', 
                '#8e44ad', '#16a085', '#27ae60', '#f39c12', '#c74350', '#6c8e7b', '#4e6477', 
                '#714a41', '#3e7250', '#D83636', '#36D865', '#9536D8', '#D8C436', '#36BDD8', 
                '#D8368E', '#5ED836', '#3D36D8', '#D86C36', '#36D89B', '#CB36D8', '#B6D836', 
                '#3687D8', '#D83657', '#36D844', '#7336D8', '#D8A236', '#36D8D2', '#D836AF', '#80D836', '#3650D8', '#D84A36', '#36D87A']
        c_dict = {}
        c_dict_str = {}
        for ii in range(len(colors)):
            c_dict[ii+1] = colors[ii]
            c_dict_str[str(ii+1)] = colors[ii]
        
        Batch_list[0].obsm['spatial'][:,[1,0]] = Batch_list[0].obsm['spatial'][:,[0,1]]
        Batch_list[1].obsm['spatial'][:,[1,0]] = Batch_list[1].obsm['spatial'][:,[0,1]]
        fig, ax = plt.subplots(1, 2, figsize=(20, 10)) # gridspec_kw={'wspace': 0.05, 'hspace': 0.1}

        _sc_0 = sc.pl.spatial(Batch_list[0], img_key=None, color=['mclust'], title=[''], palette=c_dict,
                            legend_fontsize=9, show=False, ax=ax[0], frameon=False,
                            spot_size=spot_size)
        _sc_0[0].set_title("ARI=" + str(ARI_list[0]), size=title_size)
        _sc_1 = sc.pl.spatial(Batch_list[1], img_key=None, color=['mclust'], title=[''], palette=c_dict,
                            legend_fontsize=9, show=False, ax=ax[1], frameon=False,
                            spot_size=spot_size)
        handles1, labels1 = ax[1].get_legend_handles_labels()
        handles0, labels0 = ax[0].get_legend_handles_labels()

        # Create a combined set of handles and labels
        combined_handles = list(set(handles0) | set(handles1))
        combined_labels = sorted(list(set(labels0) | set(labels1)), key=lambda x: int(x))
        # print(combined_labels)
        # Create custom legend handles with colors
        legend_handles = [mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor=c_dict_str[label], markersize=10, label=label) for label in combined_labels]

        ax[1].legend(handles=legend_handles, labels=combined_labels, ncol=3, bbox_to_anchor=(1.2, 1))
        ax[0].legend().remove()
        plt.subplots_adjust(wspace=-0.1)

        # plt.tight_layout()
        plt.savefig(os.path.join(exp_fig_dir, dataset_name + '_'.join(section_ids) + str(num_iter) + identifier + "_viz.pdf"))

    """save embedding + labels"""
    df_labels = ad_temp.obs[['original_clusters', 'mclust']]
    embed_ = ad_temp.obsm[use_key]
    np.save(os.path.join(exp_fig_dir, '_'.join(section_ids) + 'iter' + str(num_iter) + 'embedding_' + identifier + '.npy'),
            embed_)
    df_labels.to_csv(
        os.path.join(exp_fig_dir, '_'.join(section_ids) + 'iter' + str(num_iter) + 'labels_' + identifier + '.csv'))

    return ARI_list


def cal_layer_based_alignment_result(alignment, s1, s2):
    # fei added
    labels = []
    labels.extend(s1.obs['original_clusters'])
    labels.extend(s2.obs['original_clusters'])

    total = alignment.shape[0]
    res = []
    l_dict = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6}
    cnt0 = 0
    cnt1 = 0
    cnt2 = 0
    cnt3 = 0
    cnt4 = 0
    cnt5 = 0
    cnt6 = 0
    for i, elem in enumerate(alignment):
        if labels[i] == '-1' or labels[elem.argmax() + alignment.shape[0]] == '-1':
            continue
        if l_dict[labels[i]] == l_dict[labels[elem.argmax() + alignment.shape[0]]]:
            cnt0 += 1
        if abs(l_dict[labels[i]] - l_dict[labels[elem.argmax() + alignment.shape[0]]]) == 1:
            cnt1 += 1
        if abs(l_dict[labels[i]] - l_dict[labels[elem.argmax() + alignment.shape[0]]]) == 2:
            cnt2 += 1
        if abs(l_dict[labels[i]] - l_dict[labels[elem.argmax() + alignment.shape[0]]]) == 3:
            cnt3 += 1
        if abs(l_dict[labels[i]] - l_dict[labels[elem.argmax() + alignment.shape[0]]]) == 4:
            cnt4 += 1
        if abs(l_dict[labels[i]] - l_dict[labels[elem.argmax() + alignment.shape[0]]]) == 5:
            cnt5 += 1
        if abs(l_dict[labels[i]] - l_dict[labels[elem.argmax() + alignment.shape[0]]]) == 6:
            cnt6 += 1

    res.extend([cnt0 / total, cnt1 / total, cnt2 / total, cnt3 / total, cnt4 / total, cnt5 / total, cnt6 / total])
    return res


def cal_layer_based_alignment_result_mhypo(alignment, s1, s2):
    labels = []
    labels.extend(s1.obs['original_clusters'])
    labels.extend(s2.obs['original_clusters'])

    res = []
    cnt0 = 0

    for i, elem in enumerate(alignment):
        if labels[i] == labels[elem.argmax() + alignment.shape[0]]:
            cnt0 += 1

    return cnt0 / alignment.shape[0]


if __name__ == '__main__':
    pass
