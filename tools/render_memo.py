"""Renderiza docs/final/MEMO_EJECUTIVO.md a PDF A4 con Chrome headless (entorno .venv_docs, separado del de análisis).
Uso: .venv_docs/bin/python -I tools/render_memo.py <md> <pdf>"""
import subprocess, sys, tempfile
from pathlib import Path
import markdown
from pypdf import PdfReader
md, pdf = Path(sys.argv[1]), Path(sys.argv[2]).resolve()
body = markdown.markdown(md.read_text(), extensions=["tables"])
css = """@page{size:A4;margin:14mm 15mm 13mm 15mm}
body{font-family:-apple-system,'Helvetica Neue',Arial,sans-serif;font-size:9.6pt;line-height:1.34;color:#111}
h1{font-size:14.5pt;margin:0 0 4px} h2{font-size:11pt;margin:9px 0 3px;border-bottom:1px solid #ccc;padding-bottom:1px}
p{margin:3px 0} ul,ol{margin:2px 0 4px 18px;padding:0} li{margin:1px 0}
table{border-collapse:collapse;margin:3px 0 5px;font-size:9pt} th,td{border:1px solid #bbb;padding:2px 7px;text-align:left}
th{background:#f1f1ee} td:nth-child(n+2){text-align:right} code{font-size:8.6pt}"""
html = f"<!doctype html><html lang='es'><head><meta charset='utf-8'><style>{css}</style></head><body>{body}</body></html>"
with tempfile.TemporaryDirectory() as d:
    h = Path(d) / "memo.html"; h.write_text(html)
    chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--user-data-dir={d}/prof",
                    f"--print-to-pdf={pdf}", h.as_uri()], check=True, capture_output=True, timeout=120)
n = len(PdfReader(pdf).pages)
print(f"{pdf.name}: {n} páginas")
sys.exit(0 if n <= 2 else 1)
