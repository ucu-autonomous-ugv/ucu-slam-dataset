# UCU SLAM Dataset

This repository contains the data processing pipeline (calibration, cleaning, export) used to produce the **UCU SLAM Dataset**.

The dataset itself is hosted on Hugging Face:

**https://huggingface.co/datasets/ucu-autonomous-ugv/ucu-slam-dataset-v1**

All details about the dataset — sequences, sensors, format, and usage — are documented on the dataset page above.

> [!NOTE]
> **This is a gated dataset.** To access the files, log in to Hugging Face and accept the conditions on this page. Your contact information will be shared with the dataset authors. Once access is granted, authenticate locally and download as usual:
>
> ```bash
> hf auth login
> hf download ucu-autonomous-ugv/ucu-slam-dataset-v1 --repo-type dataset --include "sequence-1/*" --local-dir ucu-slam-dataset-v1
> ```

## Repository structure

```
.
├── calibration/           # Calibration targets, configs, and raw calibration recordings
├── calibration.Snakefile  # Snakemake pipeline for camera/sensor calibration
├── clean.Snakefile        # Snakemake pipeline for cleaning raw recordings
├── scripts/                # Cleaning, export (TUM format), and preview generation scripts
├── examples/               # Example usage
└── requirements.txt
```

## Citation

```bibtex
@misc{ucu_slam_dataset_v1_2026,
  title     = {{UCU SLAM Dataset v1}},
  author    = {{Andriy Kryvyi, Hordii Yeliseev, Oleksandr Kosovan, Yaroslav Prytula}},
  year      = {2026},
  publisher = {Hugging Face},
  url       = {https://huggingface.co/datasets/ucu-autonomous-ugv/ucu-slam-dataset-v1}
}
```
