# TCGA-OV-Drug-Response-Prediction
TCGA-OV Platinum Drug Response Prediction using DNA Methylation, Proteomics, Transcriptomics &amp; Early Fusion (AUC up to 0.94)
# TCGA-OV Drug Response Prediction (Multi-Omics)

Predicting platinum-based drug response in ovarian cancer (TCGA-OV) patients who received drug treatment (`pharma_yes` subgroup) using four high-performance models.

## Models Included
| Omics Type       | Best Model      | Features | AUC (internal test) |
|------------------|-----------------|----------|---------------------|
| DNA Methylation  | CatBoost        | ~2k      | 0.89                |
| Proteomics (RPPA)| XGBoost         | 192      | 0.91                |
| Transcriptome    | RandomForest    | ~1k      | 0.87                |
| Early Fusion     | RandomForest    | ~4k      | **0.94**            |

## Quick Start

```bash
git clone https://github.com/AFS-lunlun/TCGA-OV-Drug-Response-Prediction.git
cd TCGA-OV-Drug-Response-Prediction
pip install -r requirements.txt
