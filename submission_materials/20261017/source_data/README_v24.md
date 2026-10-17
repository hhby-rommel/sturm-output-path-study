# v24 source-data scope

Active manuscript matrices: ../audit/publication_data.json. Five-seed core: five_seed_per_run_metrics.csv, five_seed_mean_sample_sd.csv, five_seed_event_metrics_and_counts.csv, test_event_counts_and_extent.csv, five_seed_paired_F1_all_metrics.csv, Table_E_8a_five_seed.csv, Table_E_8b_five_seed.csv, paired_extent_summary.csv and event_extent_mean_sd.csv.

Updated companion names retain their historical filenames: v1_table_E_8.csv and v1_table_E_10.csv now label test rows as five seeds and validation rows as three seeds. test_event_extent_bias_full.csv, event_F1_extent_reconciled_perturbed.csv, Figure_3_event_differences.csv, phase1_LOEO_*.csv, phase1_EMSR551_seed_FP_FN.csv, phase1_bias_variance_decomposition.csv and phase1_ranking_per_seed_and_mean.csv use five-seed accuracy/extent data. CPU records retain their original three checkpoints.

Other old auxiliary CSVs, including phase1_ranking_summary/pairwise_flips/ties, retain their historical three-seed scope and are not substituted for the updated five-seed manuscript tables. Validation, spatial, threshold-transfer, Sen1 and hardware sources retain their original evidence scope.

Extent error is averaged after taking the absolute value per event and seed. Signed bias is reported separately. CPU values do not become five-seed measurements when displayed next to five-seed F1.
