# Deployment Notes

This project is ready for a simple Render web-service deployment.

## Files Needed

- `src/rf_shap_web_app.py`
- `src/train_rf_shap_model.py`
- `src/predict_rf_shap_agent.py`
- `models/rf_shap_all_nanocarrier_final_model.joblib`
- `outputs/rf_shap_all_nanocarrier_final_summary.json`
- `requirements.txt`
- `render.yaml`

The raw paper PDFs and source JSON extraction files are not required for the public UI.

## Render Setup

1. Upload this project to a GitHub repository.
2. In Render, create a new Web Service from that repository.
3. Render can detect `render.yaml`; otherwise use:
   - Build command: `pip install -r requirements.txt`
   - Start command: `python src/rf_shap_web_app.py`
4. The app reads Render's `PORT` environment variable automatically.

## Notes

- The free Render tier may sleep after inactivity, so the first load can be slow.
- The model is a screening/support tool, not a definitive experimental replacement.
- Keep original paper PDFs and full extraction folders out of the public repository unless your team explicitly wants to share them.
