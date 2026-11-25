"""Execute the embedding notebooks in process and save their outputs.

Run from the repository root: python scripts/run_notebooks.py
Figures are saved by the notebooks; dataframe outputs are stored as plain text.
No training is performed.
"""
from __future__ import annotations

import contextlib
import io
from pathlib import Path
import sys
import traceback

import matplotlib
import nbformat

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def display_text(value):
    print(value.to_string(index=False) if hasattr(value, "to_string") else value)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    root = Path(__file__).resolve().parents[1]
    for name in ["07_embedding_geometry.ipynb", "08_final_analysis.ipynb"]:
        path = root / "notebooks" / name
        notebook = nbformat.read(path, as_version=4)
        namespace = {"__name__": "__main__", "display": display_text}
        count = 0
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            count += 1
            print(f"{name}: cell {index} started", flush=True)
            # Do not assert a p-value before reading the actual test result.
            if name.startswith("07"):
                cell.source = cell.source.replace(
                    'f"{m.upper()} — Spearman ρ = {rho_val:.4f} (p < 1e-4)"',
                    'f"{m.upper()} — Spearman ρ = {rho_val:.4f}"',
                )
            output = io.StringIO()
            cell.execution_count = count
            try:
                with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                    exec(compile(cell.source, f"{name}:cell_{index}", "exec"), namespace)
                cell.outputs = [nbformat.v4.new_output("stream", name="stdout", text=output.getvalue())]
            except Exception as error:
                cell.outputs = [
                    nbformat.v4.new_output("stream", name="stdout", text=output.getvalue()),
                    nbformat.v4.new_output("error", ename=type(error).__name__,
                                           evalue=str(error), traceback=traceback.format_exc().splitlines()),
                ]
                nbformat.write(notebook, path)
                raise
            nbformat.write(notebook, path)
            plt.close("all")
            print(f"{name}: cell {index} complete", flush=True)


if __name__ == "__main__":
    main()
