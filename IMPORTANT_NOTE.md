# IMPORTANT NOTE — Dataset Not Included in Submission

## EC7207 — High Performance Computing Assignment
**Team:** EG/2021/4426 · EG/2021/4432 · EG/2021/4433

---

## Why the Dataset is Missing

The UNSW-NB15 dataset file used in this project exceeds the submission size limit:

| File | Size |
|------|------|
| `UNSW-NB15_1_with_header.csv` | ~157 MB |
| Maximum submission file size | 10 MB |

The dataset **cannot be included** in the submitted archive. It has been uploaded to Google Drive instead.

---

## Download the Dataset

**Google Drive Link:**
https://drive.google.com/drive/folders/1tqNgeGTsgRTTDDsr46wnJ4Gt4gdxUfzN?usp=sharing

### Setup Instructions (after downloading)

1. Download the file `UNSW-NB15_1_with_header.csv` from the Drive link above.
2. Place it in the project at this exact path:

```
data/UNSW-NB15_1.csv/UNSW-NB15_1_with_header.csv
```

Full directory structure to create:
```
HPC_Network_Analysis/
└── data/
    └── UNSW-NB15_1.csv/
        └── UNSW-NB15_1_with_header.csv   ← place file here
```

3. Run the project normally (all programs auto-detect this path):

```bash
make all
cd webapp && python3 app.py
# Open http://localhost:5000
```

---

## Notes

- All six implementations (Serial, OpenMP, Pthreads, MPI, Hybrid, CUDA) read the dataset from the path above.
- The `.gitignore` excludes `data/` and `*.csv` — this is intentional to prevent accidental commits of large files.
- All performance results, charts, and logs in `results/` and `charts/` were generated from this dataset on our test machine (VMware Ubuntu) and Google Colab (CUDA / Tesla T4).
- The dataset is the original **UNSW-NB15** network intrusion benchmark from UNSW Canberra. Records: **700,001**.

---

*This note was included because the dataset file size (157 MB) exceeds the 10 MB submission limit.*
