"""Convert the percent-format script er_kaggle.py into a Kaggle notebook (amazon_er_kaggle.ipynb)."""
import re
import sys
from pathlib import Path

import nbformat

src = Path(__file__).with_name(sys.argv[1] if len(sys.argv) > 1 else "er_kaggle.py").read_text(encoding="utf-8")
cells = []
for block in re.split(r"^# %%", src, flags=re.M)[1:]:
    header, _, body = block.partition("\n")
    if header.strip() == "[markdown]":
        md = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.strip().splitlines())
        cells.append(nbformat.v4.new_markdown_cell(md))
    else:
        cells.append(nbformat.v4.new_code_cell(body.strip("\n")))
nb = nbformat.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "language_info": {"name": "python"},
    "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True},
})
out = Path(__file__).with_name(sys.argv[2] if len(sys.argv) > 2 else "amazon_er_kaggle.ipynb")
nbformat.write(nb, out)
print(f"wrote {out} with {len(cells)} cells", file=sys.stderr)
