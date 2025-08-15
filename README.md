\# Synthetic Uruguay – supervised learning demo (FULLY SYNTHETIC)



\## Kom i gang

```bash

python -m venv .venv

\# Windows: .venv\\Scripts\\activate

source .venv/bin/activate

pip install -r requirements.txt



\# 1) Generer syntetiske data

python generator.py



\# 2) Tren modeller (lagrer joblib + metrics)

python train\_models.py --input ./data/synthetic\_uruguay\_attempts.csv --outdir ./models



\# 3) Batch-prediksjon (CSV -> CSV)

python predict.py ./data/synthetic\_uruguay\_attempts.csv \\

&nbsp; --model ./models/logreg\_pipeline.joblib \\

&nbsp; --spec  ./models/model\_spec.json \\

&nbsp; --output ./data/preds\_logreg.csv \\

&nbsp; --id-col ID\_Number \\

&nbsp; --proba-col pred\_proba \\

&nbsp; --flag-topq 0.90



\# 4) Top-K (kun de høyeste sannsynlighetene)

python predict\_topk.py ./data/synthetic\_uruguay\_attempts.csv \\

&nbsp; --model ./models/hgb\_calibrated.joblib \\

&nbsp; --spec  ./models/model\_spec.json \\

&nbsp; --id-col ID\_Number \\

&nbsp; --k 200 \\

&nbsp; --min-proba 0.35 \\

&nbsp; --output ./data/top200\_hgb.csv



\# 5) Finn terskler for presisjon/recall på holdout-året

python predict\_thresholds.py \\

&nbsp; --data ./data/synthetic\_uruguay\_attempts.csv \\

&nbsp; --model ./models/logreg\_pipeline.joblib \\

&nbsp; --spec  ./models/model\_spec.json \\

&nbsp; --out   ./models/thresholds.json \\

&nbsp; --target-precision 0.60 \\

&nbsp; --target-recall 0.50



