# Hyperbolic Barnes-Hut tSNE

- Link to data and project details: https://surfdrive.surf.nl/files/index.php/f/12451516489

## Plan for February 2023

Highest priority TODOs:
- Polish Hyperbolic tSNE code.
- Get acquainted with Hyperbolic tSNE code. Where are the different components and parameters? Specially those we want to test in the ablation experiment (splitting strategy and einstein vs frechet midpoints).
  - Define the different experimental conditions. For instance, in experiment 4.2.1 we are interested on early exaggeration only vs grad desc only.
  - Define sampling strategy for getting differently sized datasets.
- Make inventory of datasets we want to run our experiments in.
- Make note of what we want to log.
  - Per experiment: dataset, size, machine, number of runs per condition, random seed per run 
  - Per run: number of iterations, time per iteration, cost function per iteration, embedding per iteration
  - Note: logging this stuff might significantly alter times so we might need to perform two runs, one with and one without logging.
  - Some ideas: time, number of iterations, cost function, embeddings
- Write a script that runs the grid of experiments defined above.
- Run the script on a subset of the datasets and verify that it makes sense the results we are getting. Pay special focus to experiment 4.2, which has the highest priority.

Other TODOs:
- Find implementations of competing methods and run our grid of experiments with them.


## Related works

Below we list the related works. For each, we list the datasets they used and, if available, provide a link to their implementation.
We compare our method with theirs.

- [1] Guo, Y., Guo, H., & Yu, S. X. (2022). **CO-SNE: Dimensionality Reduction and Visualization for Hyperbolic Data.** In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 21-30).
- [2] Klimovskaia, A., Lopez-Paz, D., Bottou, L., & Nickel, M. (2020). **Poincaré maps for analyzing complex hierarchies in single-cell data.** Nature communications, 11(1), 1-9.
- [3] Zhou, Y., & Sharpee, T. O. (2021). **Hyperbolic geometry of gene expression.** Iscience, 24(3), 102225.

[1] CO-SNE: Dimensionality Reduction and Visualization for Hyperbolic Data
- Data:
  - Synthethic dataset of mixture of hyperbolic gaussians
  - Hematopoiesis from [2], which comes from [6]
  - Hypernymy relations of the mammals subtree of WordNet following [7]
  -  MNIST with derived features from a hyperbolic DNN and VAE
- Code: https://github.com/yunhuiguo/CO-SNE

[2] Poincaré maps for analyzing complex hierarchies in single-cell data
- Data:
  - Synthetic datasets generated with ScanPy
  - Data from [6] (synapse ID https://www.synapse.org/#Synapse: syn4975060syn4975060)
  - Data from [8] (accession code https://www.ncbi.nlm.nih.gov/geo/ query/acc.cgi?acc=GSE72857GSE72857)
  - Data from [9] (accession code https://www. ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE61470GSE61470)
  - Data from [10] (accession code https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE103633GSE103633, preprocessed data available at https://shiny.mdc-berlin.de/psca/)
  - Data from [11] (preprocessed data available at https://github.com/qinzhu/VisCello)
- Code: https://github.com/facebookresearch/PoincareMaps

[3] Hyperbolic geometry of gene expression
- Data:
  - Microarray dataset from human samples [4] (https://www.ebi.ac.uk/arrayexpress/experiments/E-MTAB-62/)
  - scRNA-seq [5] (https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE108097)
  - Hematopoiesis from [2], which comes from [6] (https://github.com/facebookresearch/PoincareMaps/blob/master/datasets/)
- Code: https://github.com/gyrheart/Hyperbolic-t-SNE


## Other references 

- [4] Lukk, M., Kapushesky, M., Nikkila ̈ , J., Parkinson, H., Goncalves, A., Huber, W., Ukkonen, E., and Brazma, A. (2010). A global map of human gene expression. Nat. Biotechnol. 28, 322.
- [5] Han, X., Wang, R., Zhou, Y., Fei, L., Sun, H., Lai, S., Saadatpour, A., Zhou, Z., Chen, H., Ye, F., et al. (2018). Mapping the mouse cell atlas by microwell-seq. Cell 172, 1091–1107.
- [6] Andre Olsson, Meenakshi Venkatasubramanian, Viren K Chaudhri, Bruce J Aronow, Nathan Salomonis, Harinder 1 Singh, and H. Leighton Grimes. Single-cell analysis of mixed-lineage states leading to a binary cell fate choice. Nature, 2016.
- [7] Maximilian Nickel and Douwe Kiela. Poincare embeddings for learning hierarchical representations. arXiv preprint arXiv:1705.08039, 2017.
- [8] Paul, F. et al. Transcriptional heterogeneity and lineage commitment in myeloid progenitors. Cell 163, 1663–1677 (2015).
- [9] Moignard, V. et al. Decoding the regulatory network of early blood development from single-cell gene expression measurements. Nat. Biotechnol. 33, 269 (2015).
- [10] Plass, M. et al. Cell type atlas and lineage tree of a whole complex animal by single-cell transcriptomics. Science 360, eaaq1723 (2018).
- [11] Packer, J. S. et al. A lineage-resolved molecular atlas of C. elegans embryogenesis at single-cell resolution. Science 365, eaax1971 (2019).
