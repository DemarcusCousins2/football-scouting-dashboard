# Football Scouting Dashboard

A Streamlit app that turns a Hudl scouting CSV into offense/defense tendency tables,
pre-snap tells, field-position and yards-allowed density charts, and a downloadable Excel report.

## Run locally

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/streamlit run main.py
```

Open the browser tab it launches and upload a Hudl scouting CSV. When running locally you can
also just drop a CSV in this folder and the app auto-loads the newest one.

## Deploy to Streamlit Community Cloud (to share a link with your coach)

1. Push this folder to a **GitHub repo** (public, or **private** if the scouting data is confidential).
2. Go to https://share.streamlit.io, sign in with GitHub, and click **New app**.
3. Select your repo, branch `main`, and main file `main.py`.
4. Under **Advanced settings**, set the Python version to **3.12** (this app targets 3.11–3.12).
5. Click **Deploy**. You'll get a link like `https://<name>.streamlit.app` to send your coach.

On the hosted app, the coach clicks **Upload a Hudl scouting CSV** to load data — the
auto-load-from-folder feature only works when running locally. Scouting CSVs/PDFs/XLSX are
gitignored by default so they aren't published; remove those lines in `.gitignore` if you want a
demo dataset baked into the app.
