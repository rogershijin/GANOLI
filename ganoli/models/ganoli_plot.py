import scanpy as sc
import anndata as ad
sc.set_figure_params(dpi=300)
sc._settings.ScanpyConfig.n_jobs = 4

def plot_umap(data, labels=None, label_name=None, pc_range='all'):
    data = ad.AnnData(data)
    if labels is not None:
        data.obs[label_name] = labels
    sc.pp.neighbors(data, n_neighbors=10, n_pcs=40)
    sc.tl.leiden(data)
    sc.tl.paga(data)
    sc.pl.paga(data, plot=False)
    sc.tl.umap(data, init_pos='paga')
    sc.pl.umap(data, color=label_name) # for atac, get rid of 1st pc) 
    return data