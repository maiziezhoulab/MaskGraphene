import logging
import numpy as np
from tqdm import tqdm
import torch
import wandb
import os
import scanpy as sc
from torch.cuda.amp import autocast, GradScaler
import gc
import pandas as pd
from sklearn.metrics import adjusted_rand_score
# from st_loading_utils import gmm

from utils import (
    build_args_ST,
    create_optimizer,
    set_random_seed,
    show_occupied_memory
)
from datasets.lc_sampler import (
    setup_training_dataloder
)
from datasets.data_proc import load_ST_dataset
from datasets.st_loading_utils import visualization_umap_spatial, create_dictionary_mnn, gmm_scikit
from models import build_model_ST


# def MG(model, dataloader, optimizer, max_epoch, device, adata_concat_, scheduler, key_="MG", logger=None):
#     logging.critical("start training..")

#     for epoch in range(max_epoch):
#         epoch_iter = tqdm(dataloader)
#         total_loss = 0
#         num_instances = 0

#         model.train()
#         optimizer.zero_grad()

#         for batch_g in epoch_iter:
#             model.train()
#             batch_g, targets, _, node_idx = batch_g
#             batch_g = batch_g.to(device)
#             x = batch_g.ndata.pop("feat")
#             loss = model(batch_g, x, targets=targets)

#             # loss_dict = {"loss": loss.item()}
#             # optimizer.zero_grad()
#             # loss.backward()
#             # optimizer.step()
#             epoch_iter.set_description(f"train_loss: {loss.item():.4f}, Memory: {show_occupied_memory():.2f} MB")
            
#             batch_size = x.shape[0]
#             total_loss += loss * batch_size
#             num_instances += batch_size
#         if scheduler is not None:
#             scheduler.step()
#         # Compute average loss and backpropagate
#         avg_loss = total_loss / num_instances
#         avg_loss.backward()
#         optimizer.step()
#     with torch.no_grad():
#         model.eval()
#         all_embeddings = []

#         for batch_g in tqdm(dataloader, desc="Embedding"):
#             batch_g, targets, _, node_idx = batch_g
#             batch_g = batch_g.to(device)
#             x = batch_g.ndata.pop("feat")

#             z = model.embed(batch_g, x)  # shape: [#nodes in batch, hidden_dim]
#             all_embeddings.append(z.cpu())

#         z_full = torch.cat(all_embeddings, dim=0)
#         adata_concat_.obsm[key_] = z_full.numpy()
    
#     return model, adata_concat_


def MG(model, dataloader, optimizer, max_epoch, device, adata_concat_, scheduler, key_="MG", logger=None, accum_steps=16):
    logging.critical("start training..")
    scaler = GradScaler()
    ari_log = []

    epoch_bar = tqdm(range(max_epoch), desc="Epochs")
    for epoch in epoch_bar:
        model.train()
        optimizer.zero_grad()
        epoch_iter = tqdm(enumerate(dataloader), total=len(dataloader))
        total_loss = 0.0
        num_instances = 0

        for i, batch in epoch_iter:
            batch_g, targets, _, node_idx = batch
            batch_g = batch_g.to(device)
            x = batch_g.ndata.pop("feat")

            with autocast():
                loss = model(batch_g, x, targets=targets)
                loss = loss / accum_steps  # normalize loss for accumulation

            scaler.scale(loss).backward()

            batch_size = x.shape[0]
            total_loss += loss.item() * accum_steps * batch_size  # reverse normalize
            num_instances += batch_size

            if (i + 1) % accum_steps == 0 or (i + 1) == len(dataloader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_iter.set_description(f"train_loss: {loss.item() * accum_steps:.4f}, Mem: {show_occupied_memory():.2f} MB")

            # Free up memory
            del batch_g, x, targets, loss
            torch.cuda.empty_cache()
            gc.collect()

        if scheduler is not None:
            scheduler.step()

        # ======= Embedding and Evaluation Every 200 Epochs =======
        if (epoch + 1) % 200 == 0 or (epoch + 1) == max_epoch:
            all_embeddings = []
            all_node_indices = []

            model.eval()
            with torch.no_grad():
                for batch_g, targets, _, node_idx in tqdm(dataloader, desc=f"Eval Epoch {epoch+1}"):
                    batch_g = batch_g.to(device)
                    x = batch_g.ndata.pop("feat")

                    z = model.embed(batch_g, x)
                    all_embeddings.append(z.cpu())
                    all_node_indices.append(node_idx)

            z_full = torch.cat(all_embeddings, dim=0)
            node_indices = torch.cat(all_node_indices, dim=0)
            z_ordered = torch.zeros_like(z_full)
            z_ordered[node_indices] = z_full

            use_key = 'temp'
            adata_concat_.obsm[use_key] = z_ordered.numpy()

            # Run clustering and compute ARI per section
            from collections import defaultdict
            section_aris = defaultdict(float)
            num_class = 15

            for section in adata_concat_.obs['batch_name'].unique():
                ad_temp = adata_concat_[adata_concat_.obs['batch_name'] == section].copy()
                ad_temp = gmm_scikit(ad_temp, num_class=num_class, used_obsm=use_key)

                pred = ad_temp.obs['mclust']
                truth = ad_temp.obs['annotation']
                ari = adjusted_rand_score(truth, pred)
                section_aris[section] = ari

            # Log ARIs
            log_msg = f"[Epoch {epoch+1}] ARIs:"
            ari_entry = {"epoch": epoch + 1}
            for sec, score in section_aris.items():
                log_msg += f" {sec}: {score:.4f}"
                ari_entry[sec] = score
            ari_log.append(ari_entry)

            if logger:
                logger.info(log_msg)
            else:
                print(log_msg)

    # ======= Embedding =======
    all_embeddings = []
    all_node_indices = []

    with torch.no_grad():
        model.eval()
        for batch_g, targets, _, node_idx in tqdm(dataloader, desc="Embedding"):
            batch_g = batch_g.to(device)
            x = batch_g.ndata.pop("feat")

            z = model.embed(batch_g, x)
            all_embeddings.append(z.cpu().detach())
            all_node_indices.append(node_idx)

    z_full = torch.cat(all_embeddings, dim=0)
    node_indices = torch.cat(all_node_indices, dim=0)

    # Reconstruct in original order
    z_ordered = torch.zeros_like(z_full)
    z_ordered[node_indices] = z_full

    adata_concat_.obsm[key_] = z_ordered.numpy()
    ari_df = pd.DataFrame(ari_log)
    return model, adata_concat_, ari_df


# def MG_triplet(model, dataloader, optimizer, max_epoch, device, adata_concat_, scheduler, logger=None, key_="MG_triplet", iter_comb=None):
#     logging.critical("Start training with Triplet Loss...")
#     section_ids = np.array(adata_concat_.obs['batch_name'].unique())

#     for epoch in tqdm(range(max_epoch), desc="Training"):
#         if epoch % 100 == 0:
#             mnn_dict = create_dictionary_mnn(adata_concat_, use_rep="MG", batch_name='batch_name', k=50, iter_comb=iter_comb)
#             anchor_ind = []
#             positive_ind = []
#             negative_ind = []
#             for batch_pair in mnn_dict.keys():  # pairwise compare for multiple batches
#                 batchname_list = adata_concat_.obs['batch_name'][mnn_dict[batch_pair].keys()]
#                 cellname_by_batch_dict = dict()
#                 for batch_id in range(len(section_ids)):
#                     cellname_by_batch_dict[section_ids[batch_id]] = adata_concat_.obs_names[
#                         adata_concat_.obs['batch_name'] == section_ids[batch_id]].values

#                 anchor_list = []
#                 positive_list = []
#                 negative_list = []
#                 for anchor in mnn_dict[batch_pair].keys():
#                     anchor_list.append(anchor)
#                     ## np.random.choice(mnn_dict[batch_pair][anchor])
#                     positive_spot = mnn_dict[batch_pair][anchor][0]  # select the first positive spot
#                     positive_list.append(positive_spot)
#                     section_size = len(cellname_by_batch_dict[batchname_list[anchor]])
#                     negative_list.append(
#                         cellname_by_batch_dict[batchname_list[anchor]][np.random.randint(section_size)])

#                 batch_as_dict = dict(zip(list(adata_concat_.obs_names), range(0, adata_concat_.shape[0])))
#                 anchor_ind = np.append(anchor_ind, list(map(lambda _: batch_as_dict[_], anchor_list)))
#                 positive_ind = np.append(positive_ind, list(map(lambda _: batch_as_dict[_], positive_list)))
#                 negative_ind = np.append(negative_ind, list(map(lambda _: batch_as_dict[_], negative_list)))
#         model.train()
#         optimizer.zero_grad()
#         losses = []
#         num_instances = 0

#         for batch_g in dataloader:
#             model.train()
#             batch_g, targets, _, _ = batch_g
#             batch_g = batch_g.to(device)
#             x = batch_g.ndata.pop("feat")

#             optimizer.zero_grad()
#             loss = model(batch_g, x, targets=targets)  # classification loss
#             # loss.backward()
#             # optimizer.step()

#             batch_size = x.shape[0]
#             losses.append(loss * batch_size)
#             num_instances += batch_size
        
#         with torch.no_grad():
#             all_embeddings = []
#             for batch_g in dataloader:
#                 batch_g, targets, _, _ = batch_g
#                 batch_g = batch_g.to(device)
#                 x = batch_g.ndata.pop("feat")
#                 z = model.embed(batch_g, x)
#                 all_embeddings.append(z.cpu())
#         z = torch.cat(all_embeddings, dim=0)

#         anchor_arr = z[anchor_ind,]
#         positive_arr = z[positive_ind,]
#         negative_arr = z[negative_ind,]

#         triplet_loss = torch.nn.TripletMarginLoss(margin=1, p=2, reduction='mean')
#         tri_output = triplet_loss(anchor_arr, positive_arr, negative_arr)

#         avg_loss = sum(losses) / num_instances
#         loss = avg_loss + tri_output
#         loss.backward()
#         optimizer.step()

#         if scheduler is not None:
#             scheduler.step()
#         # loss_dict = {"loss": loss.item()}
#         tqdm.write(f"[Epoch {epoch+1}/{max_epoch}] Total Loss: {avg_loss:.4f}, Triplet Loss: {tri_output}")
        
#         # if logger is not None:
#         #     # loss_dict["lr"] = get_current_lr(optimizer)
#         #     # logger.log(loss_dict, step=epoch)
#         #     logger.log({'epoch': epoch+1, 'ratio': r, 'layer-wise acc': lw_acc[0], 'loss':loss.item()})

#     # Final embedding
#     model.eval()
#     with torch.no_grad():
#         all_embeddings = []
#         for batch_g in dataloader:
#             batch_g, targets, _, _ = batch_g
#             batch_g = batch_g.to(device)
#             x = batch_g.ndata.pop("feat")
#             z = model.embed(batch_g, x)
#             all_embeddings.append(z.cpu())
#         z_final = torch.cat(all_embeddings, dim=0)

#     adata_concat_.obsm[key_] = z_final.numpy()

#     return model, adata_concat_


def main(args):
    device = args.device if args.device >= 0 else "cpu"
    print(device)
    seeds = args.seeds
    dataset_name = args.dataset
    max_epoch = args.max_epoch
    max_epoch_triplet = args.max_epoch_triplet
    # if os.path.exists(args.log_name):
    # args.log_name = args.log_name.split(".")[0]+time
    logging.basicConfig(filename=args.log_name, format="%(asctime)s - %(levelname)s - %(message)s", level=logging.CRITICAL)
    # num_hidden = args.num_hidden
    # num_layers = args.num_layers
    # encoder_type = args.encoder
    # decoder_type = args.decoder
    # replace_rate = args.replace_rate
    # is_consecutive = args.consecutive_prior

    optim_type = args.optimizer 
    # loss_fn = args.loss_fn

    lr = args.lr
    weight_decay = args.weight_decay
    # logs = args.logging
    logs = False
    use_scheduler = args.scheduler

    """ST loading"""
    if args.section_ids == None:
        # section_ids = ['E11.5_E1S1', 'E12.5_E1S1', 'E13.5_E1S1', 'E14.5_E1S1', 'E15.5_E1S1', 'E16.5_E1S1']
        section_ids = ['E11.5_E1S1', 'E12.5_E1S1', 'E13.5_E1S1', 'E14.5_E1S1', 'E15.5_E1S1', 'E16.5_E1S1']
        # section_ids = ['E15.5_E1S1', 'E16.5_E1S1']
        args.st_data_dir = '/maiziezhou_lab/yunfei/Projects/spatial_benchmarking/benchmarking_data/Embryo'
        args.hl_dir = '/maiziezhou_lab/yunfei/Projects/MaskGraphene_revision/embryo_multi_gpu/Embryo'
        dataset_name='Embryo'
        mini_bs = 1024 * 32 # 1024
        args.num_hidden = "512,32"
        use_scheduler=False
        max_epoch=1000
        args.hvgs=5000
        args.mask_rate=0.1
        args.remask_rate=0.1
        args.num_remasking =1
        args.alpha_l=1
        # args.attn_drop=0
        # args.in_drop=0
        # max_epoch_triplet=10
    else:
        section_ids = args.section_ids.lstrip().split(",")
    # print(section_ids)
    """file save path"""
    exp_fig_dir = './notebooks/temp'
    # st_data_dir = args.st_data_dir

    model_dir = "checkpoints"
    os.makedirs(model_dir, exist_ok=True)

    if not os.path.exists(os.path.join(exp_fig_dir, dataset_name+'_'.join(section_ids))):
        os.makedirs(os.path.join(exp_fig_dir, dataset_name+'_'.join(section_ids)))

    exp_fig_dir = os.path.join(exp_fig_dir, dataset_name+'_'.join(section_ids))

    seed = seeds[0]
    set_random_seed(seed)

    if logs:
        logger = wandb.init(name='_'.join(section_ids)+"_seed"+str(seed))
    else:
        logger = None

    logging.critical('_'.join(section_ids))

    """
    this should only be applied to embryo data due to memory issue
    """
    graph, (num_features, num_cls), ad_concat = load_ST_dataset(dataset_name=dataset_name, section_ids=section_ids, args_=args)
    args.num_features = num_features
    args.num_class = num_cls
    x = graph.ndata["feat"]

    print(args)
    logging.critical(args)
    model = build_model_ST(args)
    print(model)
    logging.critical(model)
    
    optimizer = create_optimizer(optim_type, model, lr, weight_decay)

    graph_nodes = graph.num_nodes()  # or graph.shape[0] if using a tensor-like graph object
    print(graph_nodes)
    # Fake ego_graph_nodes: each node is its own subgraph (for simplicity)
    ego_graph_nodes = [np.array([i]) for i in range(graph_nodes)]
    dataloader = setup_training_dataloder(
        'lc', ego_graph_nodes, graph, x, batch_size=mini_bs, drop_edge_rate=0)

    if use_scheduler:
        logging.critical("Use scheduler")
        scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epoch) ) * 0.5
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
    else:
        scheduler = None

    model.to(device)
    model, ad_concat_1, temp_aris = MG(model, dataloader, optimizer, max_epoch, device, ad_concat, scheduler, logger=logger, key_="MG")
    # model, ad_concat_2 = MG_triplet(model, dataloader, optimizer, max_epoch_triplet, device, adata_concat_=ad_concat_1, scheduler=scheduler, logger=logger, key_="MG_triplet")
    print(temp_aris)
    ari_ = visualization_umap_spatial(ad_temp=ad_concat_1, section_ids=section_ids, exp_fig_dir='/maiziezhou_lab/yunfei/Projects/MaskGraphene_revision/embryo_multi_gpu/out', dataset_name=dataset_name, num_iter=0, identifier="stage2", num_class=args.num_class, use_key="MG", args=args)
    print(ari_)
    # counter += 1
    # if logger is not None:
    #     logger.log({"slice1_ari_after": ari_[0], "slice2_ari_after": ari_[1]})
    # for i in range(len(section_ids)):
    #     print(section_ids[i], ', ARI = %01.3f' % ari_[i])
    #     logging.critical(section_ids[i] + "_ari_after :" + str(ari_[i]))
    
    # if logger is not None:
    #     logger.finish()

    return None


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    args = build_args_ST()
    a_ari = main(args)
