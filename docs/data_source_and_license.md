# Data source and license audit

## Official upstream repository

- Repository: https://github.com/STURM-WEO/STURM-Flood
- Audited repository commit: `c39d9bf8ba76302663c097ec4e905c918e1e131d`
- The upstream README identifies the STURM-Flood dataset through Zenodo. Its
  resource table lists `10.5281/zenodo.12748982`; the README DOI badge also
  links `10.5281/zenodo.12748983`. This identifier discrepancy is preserved
  rather than silently choosing an unverified record.
- The project uses the local copy at
  `data/raw/Dataset`. No SAR-optical paired derivative is constructed.

## Local data evidence

- Source backup: `D:\BaiduNetdiskDownload\Dataset`.
- Project copy and integrity date: 2026-09-15.
- File counts and source/project copy checks are recorded in
  `docs/data_integrity_20260915.md`.
- Project copy: 48,556 files including two metadata CSV files, 48,554
  GeoTIFF files, 21,602 S1 image/mask tiles, and 2,675 S2 image/mask tiles.
- The project-level `data/raw/Dataset.zip` is not used as the active source;
  no archive checksum is asserted for it.
- Metadata SHA-256 values are recorded in the integrity report.

## Locked preprocessing protocol

- S1 input: `128x128x2`, `float32`; S2 input: `128x128x9`, `float64`.
- Raw mask codes observed globally: `0,1,2,3,4,5`.
- Following the official upstream `preprocess_mask` implementation, the
  operational mapping is `raw_mask == 0 -> 0` and `raw_mask > 0 -> 1`.
- Pixels outside the finite image region or equal to a finite image-declared
  nodata value are ignored in training loss and evaluation.
- Full audit and event protocol evidence: `docs/data_audit_20260915.md`.

## Attribution and license

- STURM-Flood repository license: CC BY-SA 4.0.
- Cite Notarangelo, Wirion, and van Winsen (2025), “STURM-Flood: A curated
  dataset for deep learning-based flood extent mapping leveraging Sentinel-1
  and Sentinel-2 imagery,” Big Earth Data,
  https://doi.org/10.1080/20964471.2025.2458714.
- Sentinel-1/Sentinel-2 source attribution and Copernicus EMS attribution are
  retained from the upstream repository README.
