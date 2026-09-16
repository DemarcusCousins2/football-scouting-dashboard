
## Run the scouting report

Create the project environment and install dependencies:

```bash
/opt/homebrew/bin/python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Run the report:

```bash
.venv/bin/python main.py
```

The generated heatmap is saved as `scouting_report_heatmap.png`.
