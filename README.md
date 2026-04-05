## Credit Card Fraud Detection: A Bayesian and Non-Bayesian Comparison

Credit card fraud detection with a 2x2 comparison of Bayesian and non-Bayesian methods, developed for TU Delft EE4685 Bayesian Machine Learning.

## Project context

This project is Assignment 2 for EE4685 Bayesian Machine Learning at Delft University of Technology, Q3 2025-2026. Authors: Adam El Haddouchi (5476526) and Naufal El Khatibi (5315778), MSc Electrical Engineering.

## The problem

The ULB credit card fraud dataset (Dal Pozzolo et al., 2015) contains 284,807 transactions made by European cardholders over two days in September 2013. Only 492 transactions (0.172%) are fraudulent, giving an imbalance ratio of roughly 577:1. Features V1 through V28 are the result of a proprietary PCA transformation; the original variables are unknown. The only raw features are `Time` and `Amount`.

Fraud detection is a natural setting for comparing Bayesian and non-Bayesian methods. The cost of missing a fraud far exceeds the cost of a false alarm, so knowing how confident a model is matters for operational decisions. Bayesian models provide this uncertainty directly, while non-Bayesian models do not. At the same time, confirmed fraud labels arrive with delays in real systems, making unsupervised (anomaly detection) approaches worth evaluating alongside supervised ones.

## Method: the 2x2 design

The comparison crosses two axes (supervised vs. anomaly detection, Bayesian vs. non-Bayesian) to isolate their effects:

|                    | Non-Bayesian              | Bayesian                          |
|--------------------|---------------------------|-----------------------------------|
| **Supervised**     | Logistic Regression (LR)  | Bayesian Logistic Regression (BLR)|
| **Anomaly Detection** | One-Class SVM (OC-SVM) | Bayesian Gaussian Mixture (BGMM)  |

The supervised pair shares the same likelihood and differs only in inference method (point estimate vs. Laplace approximation). The anomaly detection pair shares the same training data (normal transactions only) and differs in modeling approach. Any performance difference within a pair is due to the Bayesian treatment, not to differences in training data or model family.

## Key contributions

1. **2x2 comparison design.** Crossing supervised/anomaly with Bayesian/non-Bayesian cleanly separates the effect of Bayesian inference from the effect of labeled data.
2. **Three-bucket Bayesian decision protocol.** BLR's posterior uncertainty routes transactions into auto-approve, human-review, or auto-flag buckets, giving analysts a structured workload split that binary classifiers cannot provide.
3. **PCA geometry limitation analysis.** The proprietary PCA transformation in the dataset affects anomaly detection geometry. This is framed as a limitation rather than a finding: distance and density based methods operate in an unknown rotated space.

## Key findings

On the test set (56,962 transactions, 98 fraud), BLR achieves the highest AUPRC at 0.712, followed by LR at 0.702, BGMM at 0.691, and OC-SVM at 0.334. At the F1-optimal threshold, BLR and LR produce identical classification results (F1 = 0.819, 79 TP, 16 FP, 19 FN). The Bayesian treatment does not improve point predictions in the supervised setting.

BGMM, trained without any fraud labels, reaches F1 = 0.754 (72 TP, 21 FP, 26 FN), competitive with the supervised models. The gap between OC-SVM and BGMM (F1 0.431 vs. 0.754) is larger than the gap between BGMM and the supervised pair.

At matched probability thresholds (0.5), binary LR and the three-bucket protocol catch the same 89 of 98 test fraud cases. The protocol's value is structural, not metric-based: it routes 92.6% of transactions to auto-approve, 5.2% to human review (capturing 80 fraud), and 2.1% to auto-flag (9 fraud), concentrating analyst effort on the 7.3% of transactions most likely to need it.

See `report/main.tex` for the full discussion.

## Repository structure

```
cc-fraud-detection/
├── README.md
├── requirements.txt
├── data/
│   └── creditcard.csv          # not committed (see Setup)
├── notebooks/
│   └── Notebook.ipynb          # main analysis notebook
├── figures/                    # all generated plots
├── report/
│   ├── main.tex                # LaTeX report source
│   ├── references.bib
│   └── figures/
├── presentation/               # slide deck
└── workflow/                   # internal coordination files
```

## Setup and reproduction

### 1. Dataset

The dataset is not committed due to its size. Download it from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it at `data/creditcard.csv`.

### 2. Environment

```bash
python3 -m venv bml-project
source bml-project/bin/activate
pip install -r requirements.txt
```

### 3. Run the notebook

Open in Jupyter and run all cells, or execute from the command line:

```bash
jupyter nbconvert --to notebook --execute notebooks/Notebook.ipynb \
    --output Notebook.ipynb --ExecutePreprocessor.timeout=1800
```

Expect 5 to 10 minutes on a modern laptop. BGMM training dominates the runtime (roughly 2 to 3 minutes, hardware-dependent).

### 4. Compile the report

```bash
cd report
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

### 5. Reproducibility

`random_state=42` is used throughout. Results should be identical across runs on the same hardware. Training times may vary.

## Requirements

Dependencies are pinned in `requirements.txt`. The main packages are:

- numpy, pandas, scipy
- scikit-learn (models, preprocessing, metrics)
- matplotlib, seaborn (plotting)
- torch (used during development, not required for the final four models)

Python 3.12.9 was used during development.

## Authors

Adam El Haddouchi (5476526) and Naufal El Khatibi (5315778), TU Delft, MSc Electrical Engineering.

## Acknowledgments

The dataset was created by the Machine Learning Group at Universite Libre de Bruxelles (Dal Pozzolo et al., 2015). This project was developed for EE4685 Bayesian Machine Learning at TU Delft. No explicit license; course project.
