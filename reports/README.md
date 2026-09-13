# Saved report artifacts

The checked-in tables and figures are generated from deterministic synthetic OHLCV bars for pipeline verification and report rendering. They are not empirical market evidence and must not be used to claim predictive performance or tradability.

Regenerate verification tables with:

```powershell
& .\.venv\Scripts\python.exe scripts\generate_verification_artifacts.py
```

Render figures from existing tables without rerunning models:

```powershell
& .\.venv\Scripts\python.exe -m alphaforge.reporting
```
