# Output files

## Main tables

`table1_data_summary.csv` — data and model scale. Counts of accessions, SNPs, blocks, graph edges, phenotype summary and pretraining settings.

`table2_prediction.csv` — cross-validation and external validation metrics for seven kernel models. CV values are means over 30 repeats; MAE is in days. External metrics use the 29 accessions described in `data_provenance.md`.

`table3_known_genes.csv` — the 15 known dormancy-related genes with the smallest raw block percentile. Percentiles are positions in the genome-wide ranking of 8,589 blocks (1 = strongest). P values are block-level Satterthwaite approximations.

`table4_candidates.csv` — ten candidate blocks from the raw block kernel plus block 7960 (DOG1) and block 2106 (RGA1) as positive controls. Columns include the block P value, BH q value, lead SNP, lead-SNP t-test P value, minor allele frequency, genotype counts, median phenotype by genotype, and genes in the block.

`tableS1_sensitivity.csv` — block scan results after rank-normal transformation and after adding a residual kinship term (δ = 0.1).

`tableS2_grouped_cv.csv` — grouped cross-validation with k = 6, 8 and 10 genetic subpopulations.

## JSON and NPZ summaries

| File | Content |
|---|---|
| `prediction_metrics.json` | per-model means, standard deviations and per-repeat correlations |
| `prediction_model_comparison.csv` | paired differences and Wilcoxon P values against SNP-GBLUP and the dual kernel |
| `prediction_permutation.json` | phenotype-permutation null for the dual kernel |
| `prediction_cv_predictions.npz` | cross-validated predictions used in the main figure |
| `external_validation.json` | external Pearson, Spearman, MAE and bootstrap interval |
| `external_validation_predictions.npz` | external predictions and observed values |
| `ssl_control.json` | random-initialization control and kernel similarities |
| `grouped_cv.json` | grouped CV metrics and cluster sizes |
| `block_association_pc5.csv` | all 8,589 blocks: raw, graph-smoothed and embedding statistics |
| `block_association_pc5_summary.json` | minima, lambda_GC, FDR counts, rotation FWER and DOG1 details |
| `block_scan_sensitivity.json` | minimum P values and DOG1 ranks under each sensitivity setting |
| `block_aggregation_sensitivity.json` | block-kernel prediction after merging adjacent blocks |
| `known_gene_enrichment.json` | mean percentiles and top-set counts for 58 known genes |
| `known_dormancy_gene_ranks.csv` | per-gene percentiles and P values for all three kernels |
| `gene_block_map.csv` | mapping of all 27,445 genes to blocks |
| `candidate_regions.csv` | detailed rows for candidate and positive-control blocks |
| `snp_scan_pc5_summary.json` | SNP-level scan calibration and top hits |
| `snp_scan_pc5_pvalues.npz` | compact SNP-level P values |
| `snp_scan_pc5_top1000.csv` | top 1,000 SNP-level results |
| `qc_summary.json` | MAF, LD pruning, PCA, kinship and GWAS sanity checks |
| `annotation_summary.json` | SNP annotation counts and distance distribution |
| `accession_pcs.csv` | PC1–PC5 coordinates for the 83 accessions |
| `ld_decay_pairs.csv` | binned LD decay estimates |
| `block_pc5_rotation_focus.csv` | rotation P values for selected blocks |
| `baseline_prediction.json` | GBLUP and random-forest baseline from the initial QC |

## Figures

`scripts/12_final_figures.py` writes six figures to `results/figures/main/`:

1. workflow diagram
2. phenotype, PCA, LD decay and block span
3. prediction performance and external validation
4. block-level QQ and Manhattan plots
5. known-gene percentiles and permutation null
6. local association around the candidate region and DOG1

The analysis scripts also write diagnostic figures to `results/figures/`; these are not part of the main figure set.
