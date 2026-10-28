# Haplotype block graph for Arabidopsis seed dormancy

This repository contains the analysis code, processed genotype matrices and result tables for a study of seed dormancy in *Arabidopsis thaliana*. The phenotype is DSDS50, the number of days of dry storage required for 50% germination. The panel contains 83 natural accessions with both genotype and phenotype data.

The analysis has three parts:

1. Haplotype blocks are defined in a larger panel of 1,307 accessions, and blocks are connected by chromosome adjacency and linkage disequilibrium to form a graph.
2. A small graph Transformer is pretrained on the 1,307 accessions with a masked block reconstruction task. The encoder never sees phenotypes.
3. Phenotype prediction uses kernel ridge regression / GBLUP with a SNP kernel, a block kernel and a graph kernel. Block-level association uses a fixed-effects null (intercept + PC1–5) with a rotation-based calibration.

The main prediction model is the equal-weight dual kernel `0.5 * K_SNP + 0.5 * K_block`. The graph kernel is reported as an ablation and does not replace the block kernel.

## Data

The raw genotype table is the AtPolyDB `call_method_75` release in TAIR9 coordinates:

- Repository: <https://github.com/Gregor-Mendel-Institute/atpolydb>
- File: `call_method_75.tar.gz` (92,088,117 bytes), containing `call_method_75/call_method_75_TAIR9.csv`
- Content: 214,051 SNPs × 1,307 accessions, A/C/G/T calls, no missing genotypes

The phenotype file `data/raw/atpolydb/phenotype_published_raw.tsv` is included. The DSDS50 values are the `8_Seed Dormancy` column, originally from Atwell et al. (2010). The 29 external accessions are the non-missing `279_DSDS50` entries that have no `8_Seed Dormancy` value. The two columns are related but come from different experiments; the external analysis is exploratory.

Gene models are TAIR10/Araport11. The source GFF3 and the derived `genes_tair10.csv` are included in `data/annotations/`. SNP positions are TAIR9; no formal liftover was applied. The consistency between the two coordinate sets is checked in `scripts/03_annotate_snp_genes.py` and described in `docs/data_provenance.md`.

The 1001 Genomes matrix was used only for a population-structure cross-check. It is not required for any result in this repository.

## Repository layout

```
scripts/                 analysis scripts, numbered in run order
data/annotations/        TAIR10/Araport11 gene models
data/processed/          processed matrices and model artefacts
data/raw/atpolydb/       phenotype table and download notes
results/tables/          result tables, JSON summaries and numbered tables
docs/                    data provenance and output descriptions
```

Large intermediate files are not included:

- `genotypes_all_1307.npy` (267 MB) can be rebuilt with `scripts/01_build_genotype_matrix.py`.
- `block_features.npy` (172 MB, 1,307 × 8,589 × 4) can be rebuilt with `scripts/05_build_haplotype_graph.py`.

Two processed files are distributed as assets of the [latest release](https://github.com/TTFriday/arabidopsis-dormancy-graph/releases/latest) because they exceed the 25 MB limit of the GitHub web uploader:

- `graph_embeddings_83.npy` (44 MB)
- `snp_annotation.csv` (25 MB)

Run the download helper once before the analysis:

```bash
python scripts/00_download_large_files.py
```

It places both files in `data/processed/` and verifies their SHA-256 checksums. They can also be downloaded manually from the latest release page and moved to that directory.

The repository includes the 83-accession and external-29 subsets so that prediction, association and candidate-region scripts can run without the full matrices; the two release assets above complete the processed data. The pretraining script and its random-initialization control still need the full files, because they use all 1,307 accessions.

## Environment

Python 3.10 with:

```
pip install -r requirements.txt
```

The reported runs used NumPy 1.26, pandas 2.3, SciPy 1.15, scikit-learn 1.6, statsmodels 0.15, PyTorch 2.6 and Matplotlib 3.9. Training and cross-validation were run on CPU; no GPU is required.

## Running the analysis

### From the included processed data

The following scripts reproduce the main tables and figures without downloading the raw genotype file:

```bash
python scripts/00_download_large_files.py      # fetch release assets if absent
python scripts/02_qc_population_structure.py      # QC, PCA, LD decay, kinship
python scripts/04_baseline_prediction.py          # optional GBLUP / random-forest baseline
python scripts/07_prediction_models.py            # SNP, block, graph, dual and triple kernels
python scripts/07c_block_aggregation.py           # block-size sensitivity
python scripts/08_external_validation.py          # 29 external accessions
python scripts/08b_grouped_cv.py                  # grouped cross-validation
python scripts/09a_snp_scan_pc5.py                # SNP-level PC1-5 scan
python scripts/09b_block_scan_pc5.py              # block-level scan and rotation null
python scripts/09c_block_scan_sensitivity.py      # phenotype transform / kinship sensitivity
python scripts/10_biological_validation.py        # known-gene ranking and enrichment
python scripts/11_candidate_regions.py            # candidate block details
python scripts/12_final_figures.py                # main figures
python scripts/13_result_tables.py            # numbered tables
```

`02_qc_population_structure.py` also runs the PC3, EMMAX and LOCO-EMMAX sanity scans. On a laptop CPU the full set takes roughly one hour; the prediction script accounts for most of the time.

### From the raw data

1. Download `call_method_75.tar.gz` from the AtPolyDB repository and unpack it to `data/raw/atpolydb/call_method_75/`.
2. Run the pipeline. The TAIR10 GFF3 and the derived gene table are already in `data/annotations/`.

```bash
python scripts/01_build_genotype_matrix.py
python scripts/02_qc_population_structure.py
python scripts/03_annotate_snp_genes.py
python scripts/05_build_haplotype_graph.py
python scripts/06_self_supervised_pretrain.py
python scripts/06b_ssl_control.py
python scripts/07_prediction_models.py
python scripts/07c_block_aggregation.py
python scripts/08_external_validation.py
python scripts/08b_grouped_cv.py
python scripts/09a_snp_scan_pc5.py
python scripts/09b_block_scan_pc5.py
python scripts/09c_block_scan_sensitivity.py
python scripts/10_biological_validation.py
python scripts/11_candidate_regions.py
python scripts/12_final_figures.py
python scripts/13_result_tables.py
```

All scripts write to `data/processed/`, `results/tables/` and `results/figures/`. Paths are resolved relative to the repository root, so the checkout can be placed anywhere.

The pretraining script takes about 17 minutes on a 16-core CPU with single-threaded BLAS. Setting `OMP_NUM_THREADS=1` is recommended when reproducing the reported run:

```bash
OMP_NUM_THREADS=1 python scripts/06_self_supervised_pretrain.py
```

## Key result files

| File | Content |
|---|---|
| `results/tables/table1_data_summary.csv` | data and model scale |
| `results/tables/table2_prediction.csv` | cross-validation and external metrics |
| `results/tables/table3_known_genes.csv` | top 15 known dormancy genes by block percentile |
| `results/tables/table4_candidates.csv` | top candidate blocks, with DOG1 and RGA as positive controls |
| `results/tables/tableS1_sensitivity.csv` | block-association sensitivity analysis |
| `results/tables/tableS2_grouped_cv.csv` | grouped cross-validation |
| `results/tables/prediction_metrics.json` | full prediction metrics and per-repeat correlations |
| `results/tables/block_association_pc5.csv` | block-level statistics for all 8,589 blocks |
| `results/tables/known_dormancy_gene_ranks.csv` | percentiles and P values for 58 known genes |

## Notes on interpretation

- The panel has 83 phenotyped accessions. The block-level scan is calibrated but has limited power; the smallest block P value does not reach experiment-wide significance after rotation calibration.
- The graph kernel is highly correlated with the block kernel, and a randomly initialized encoder performs almost as well. The self-supervised representation is therefore reported as an ablation rather than as a source of major predictive gain.
- The external accessions have phenotypes from a different experiment and are used only as an exploratory check of rank transfer.
- SNP positions are TAIR9 and gene models are TAIR10/Araport11. Key loci were checked manually, but a formal liftover was not performed.

## Contact

Please open an issue in this repository for questions about the code or data.
