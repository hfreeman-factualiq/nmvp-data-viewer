# NMVP Data Viewer

Precomputed SharePoint discovery analytics for AINMVP1.

**Live app:** https://nmvp-data-viewer.streamlit.app

Analysis is run locally; this repo ships the UI + parquet snapshots only (no live SharePoint credentials).

### Refresh data

From the parent workspace:

```bash
python export_sharepoint_discovery_cloud.py
# then copy streamlit_cloud_sharepoint/{app.py→streamlit_app.py, data/} into this repo and push
```
