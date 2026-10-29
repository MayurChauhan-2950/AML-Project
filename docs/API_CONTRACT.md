# API contract (frontend <-> backend)

Base URL (dev): `http://127.0.0.1:8000`. The Vite dev server proxies `/api` to it, so the frontend always calls relative `/api/...`.

Classes (confirm the exact order from the saved model / notebook; alphabetical from `image_dataset_from_directory`):
`glioma`, `meningioma`, `notumor`, `pituitary`

Display labels: Glioma, Meningioma, No Tumor, Pituitary Tumor.

Medical disclaimer (use this exact text everywhere it is shown):
> AI-generated prediction for educational and research use only. It is not a medical diagnosis and must not replace review by a qualified clinician.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | `{status, model_loaded, model_version, classes}` |
| GET | `/api/models` | model comparison metrics read from `model_comparison_results.csv` (accuracy, precision, recall, F1, parameters for custom CNN and ResNet50) |
| POST | `/api/analyze` | multipart/form-data, field `file` (png/jpg/jpeg, max 10 MB). Query: `model=resnet50` (default) or `custom_cnn`, `heatmap=true|false` (default true). Returns an Analysis. |
| GET | `/api/analyses` | query `search`, `label`, `page` (1-based), `page_size`. Returns `{items:[AnalysisSummary], total, page, page_size}` |
| GET | `/api/analyses/{id}` | one Analysis |
| GET | `/api/analyses/{id}/image` | the uploaded MRI (image bytes) |
| GET | `/api/analyses/{id}/heatmap` | the Grad-CAM overlay (PNG bytes), 404 if none |
| GET | `/api/analyses/{id}/report` | PDF report (application/pdf) |
| DELETE | `/api/analyses/{id}` | delete a record and its files |
| GET | `/api/stats` | `{total_scans, last_7_days, by_class:{glioma:n,...}, average_confidence}` |

## Objects

Analysis:
```json
{
  "id": "a1b2c3d4",
  "created_at": "2026-10-29T18:30:00Z",
  "filename": "scan_01.jpg",
  "predicted_class": "glioma",
  "predicted_label": "Glioma",
  "confidence": 0.91,
  "probabilities": {"glioma": 0.91, "meningioma": 0.05, "notumor": 0.03, "pituitary": 0.01},
  "model_version": "resnet50-v1",
  "image_url": "/api/analyses/a1b2c3d4/image",
  "heatmap_url": "/api/analyses/a1b2c3d4/heatmap",
  "disclaimer": "AI-generated prediction ..."
}
```
AnalysisSummary: `id, created_at, filename, predicted_class, predicted_label, confidence`.

Errors: JSON `{"detail": "human readable message"}` with proper status codes (400 invalid file, 413 too large, 415 wrong type, 404 missing, 503 model not loaded).

## Ownership rules for parallel work
- `backend/**`: backend agent only.
- `frontend/src/layout/**`, `frontend/src/components/ui/**`, `frontend/src/styles/global.css`, `frontend/src/pages/Dashboard.jsx`, `frontend/src/pages/Settings.jsx`: shell agent only.
- `frontend/src/pages/NewAnalysis.jsx`, `Results.jsx`, `History.jsx`, `Report.jsx`, `frontend/src/components/analysis/**`, `frontend/src/lib/**`: analysis-flow agent only.
- `frontend/src/styles/tokens.css`, `frontend/src/App.jsx`, `frontend/src/main.jsx`, `frontend/package.json`, `frontend/vite.config.js`, `docs/**`: set by the lead; agents ask before changing (add dependencies with `npm install <pkg>` only if truly needed and say so in the final report).
- Nobody edits `Brain_Tumor_Classification_AML.ipynb`, `*.keras`, `model_comparison_results.csv`.
- Nobody runs `git add/commit/push/checkout`; read-only git commands only.
