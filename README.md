<h1> MaskGraphene: an advanced framework for interpretable latent representation for multi-slice, multi-condition spatial transcriptomics </h1>

Implementation for paper:  [MaskGraphene]().
<img src="/figs/pipeline.png">

<h2>Online documentation </h2>

Please refer to this [page](https://maskgraphene-tutorial.readthedocs.io/en/latest/). We hope this could help to better explore and investigate this tool.

<h2>Dependencies </h2>

* Python >= 3.9
* [Pytorch](https://pytorch.org/) == 2.0.1
* anndata==0.9.2
* h5py==3.9.0
* hnswlib==0.7.0
* igraph==0.10.8
* matplotlib==3.6.3
* paste-bio==1.4.0
* POT==0.9.1
* rpy2==3.5.14
* scanpy==1.9.1
* umap-learn==0.5.4
* wandb
* pyyaml == 5.4.1

<h2>Installation</h2>

```bash
conda create -n MaskGraphene python=3.9 

conda activate MG

git clone https://github.com/OliiverHu/MaskGraphene.git

pip install -r requirements.txt

conda install r-mclust
```

For DGL package, please refer to [link](https://www.dgl.ai/pages/start.html)

```bash
pip install  dgl -f https://data.dgl.ai/wheels/cu117/repo.html
pip install  dglgo -f https://data.dgl.ai/wheels-test/repo.html
```

<h2>Quick Start </h2>

For quick start, you could run the scripts: 

<!-- **mouse Hypothalamus -0.19/-0.24 generate hard-links**

```bash
python ../localMG_main.py --max_epoch 3000 --max_epoch_triplet 1000 --logging False --section_ids " -0.19,-0.24" --num_class 8 --load_model False --num_hidden "512,32" 
                          --exp_fig_dir "./" --h5ad_save_dir "./" --st_data_dir "./" --alpha_l 3 --lam 1 --loss_fn "sce" --mask_rate 0.50 --in_drop 0 --attn_drop 0 --remask_rate 0.50
                          --seeds 2023 --num_remasking 1 --hvgs 0 --dataset mHypothalamus --consecutive_prior 1 --lr 0.001
```

**mouse Hypothalamus -0.19/-0.24**

```bash
python ../maskgraphene_main.py --max_epoch 3000 --max_epoch_triplet 1000 --logging False --section_ids " -0.19,-0.24" --num_class 8 --load_model False --num_hidden "512,32" 
                               --exp_fig_dir "./" --h5ad_save_dir "./" --st_data_dir "./" --alpha_l 3 --lam 1 --loss_fn "sce" --mask_rate 0.50 --in_drop 0 --attn_drop 0 --remask_rate 0.50
                               --seeds 2023 --num_remasking 1 --hvgs 0 --dataset mHypothalamus --consecutive_prior 1 --lr 0.001 -->
```

**DLPFC 151507/151508 generate hard-links**

```bash
name="DLPFC"
num_layer="7"
data_path="../../spatial_benchmarking/benchmarking_data/DLPFC12"
output_path="../hard_links/DLPFC"
pairs_to_run=("151507,151508")
rounds_to_run=("1")
seeds_to_run=("2024")

for ((i=0; i<${#rounds_to_run[@]}; i++)); do
    seed=${seeds_to_run[$i]}
    round=${rounds_to_run[$i]}
    echo "Running for round $round"
    new_directory="$output_path"
    if [ ! -d "$new_directory" ]; then
        # Create the directory if it doesn't exist
        mkdir -p "$new_directory"
        echo "Created new directory: $new_directory"
    fi 
    for pair in "${pairs_to_run[@]}"; do
        echo "Running for $pair"
        python ../local_alignment_main.py --max_epoch 2000 --max_epoch_triplet 1000 --logging False --section_ids "$pair" \
                       --num_class $((num_layer)) --load_model False --num_hidden "1024,40" --alpha_l 2 --lam 1 --loss_fn "sce" --mask_rate 0.5 --in_drop 0 --attn_drop 0 --remask_rate 0.1 --exp_fig_dir "$new_directory" --h5ad_save_dir "./" --st_data_dir "$data_path" \
                       --seeds $((seed)) --sim_dir "$sim_path"  --num_remasking 1 --hvgs 8000 --dataset "$name" --consecutive_prior 1 --lr 0.001 --device 0
    done
done
```   

**DLPFC 151507/151508**

```bash
python ../maskgraphene_main.py --max_epoch 2000 --max_epoch_triplet 500 --logging False --section_ids "151507,151508" --num_class 7 --load_model False --num_hidden "512,32" 
                               --exp_fig_dir "./" --h5ad_save_dir "./" --st_data_dir "./" --alpha_l 1 --lam 1 --loss_fn "sce" --mask_rate 0.5 --in_drop 0 --attn_drop 0 --remask_rate 0.1 
                               --seeds 2023 --num_remasking 1 --hvgs 3000 --dataset DLPFC --consecutive_prior 1 --lr 0.001
```

Open-source ST datasets information:

| ST Dataset                                     | ST Type         | Abbreviations | ST Protocol   | Num. of Slices | Data Source | Annotation Source | Download Link |
|-----------------------------------------------|-----------------|---------------|---------------|----------------|-------------|-------------------|---------------|
| Human Dorsal Lateral Prefrontal Cortex (DLPFC) | Sequencing-based | DLPFC        | 10x Visium    | 12             | [spatialLIBD](http://spatial.libd.org/spatialLIBD/) | [spatialLIBD](http://spatial.libd.org/spatialLIBD/) | [Link](https://zenodo.org/records/10698880) |
| Mouse Brain Section 2 Sagittal Anterior and Posterior | Sequencing-based | MB2SA&P      | 10x Visium    | 2              | [Mouse Brain Serial Section 2 Sagittal Anterior](https://www.10xgenomics.com/resources/datasets/mouse-brain-serial-section-2-sagittal-anterior-1-standard) | [ConGI Analysis Data](https://github.com/biomed-AI/ConGI) | [Link](https://zenodo.org/records/10698931) |
| MOSTA Embryo                                   | Sequencing-based | Embryo       | Stereo-seq    | ≥50            | [MOSTA Resource](https://db.cngb.org/stomics/mosta/resource/) | [MOSTA Resource](https://db.cngb.org/stomics/mosta/resource/) | [Link](https://zenodo.org/records/10698963) |
| Mouse Hypothalamus                             | Imaging-based   | MHypo         | MERFISH       | 5              | [Datadryad](https://datadryad.org/stash/dataset/doi:10.5061/dryad.8t8s248) | [BASS Analysis Data](https://github.com/zhengli09/BASS-Analysis/blob/master/data/MERFISH_Animal1.RData) | [Link](https://zenodo.org/records/10698909) |
| Mouse Brain                                    | Imaging-based   | MB            | MERFISH       | 33             | [Zenodo Records](https://zenodo.org/records/8167488) | [Zenodo Records](https://zenodo.org/records/8167488) | [Link](https://zenodo.org/records/8167488) |                                              | [Link](https://zenodo.org/records/8167488)                                                          |


<h2>Notebook Tutorials </h2>

[Tutorial 1: hard-links generation](https://github.com/maiziezhoulab/MaskGraphene/blob/main/Tutorial%201_Hard%20link%20generation.ipynb)

[Tutorial 2: MG on DLPFC](https://github.com/maiziezhoulab/MaskGraphene/blob/main/Tutorial%202_MaskGraphene%20on%20DLPFC.ipynb)

[Tutorial 3: MG on MHypo](https://github.com/maiziezhoulab/MaskGraphene/blob/main/Tutorial%203_MaskGraphene%20on%20MHypo.ipynb)

[Tutorial 4: reproduction](https://github.com/maiziezhoulab/MaskGraphene/blob/main/Tutorial%204_Analyses%20reproduction.ipynb)


<h2> Citing </h2>



<!-- BibTeX: -->
```bibtex
@article{hu2024maskgraphene,
  title={MaskGraphene: an advanced framework for interpretable latent representation for multi-slice, multi-condition spatial transcriptomics},
  author={Yunfei Hu, Zhenhan Lin, Manfei Xie, Weiman Yuan, Yikang Li, Mingxing Rao, Yichen Henry Liu, Wenjun Shen, Lu Zhang, and Xin Maizie Zhou},
  journal={bioRxiv},
  pages={2024--02},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}