"""
Convert Final_Project_Report.md to a self-contained HTML file
with proper academic styling, then open in browser for PDF print.
"""
import os
import re
import base64
import markdown

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "Final_Project_Report.md"), "r") as f:
    md_text = f.read()

html_body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])

# Embed all images as base64 data URIs so the HTML is self-contained
def embed_images(html, base_dir):
    def replacer(match):
        src = match.group(1)
        if src.startswith(("http", "data:")):
            return match.group(0)
        abs_path = os.path.join(base_dir, src) if not os.path.isabs(src) else src
        if os.path.exists(abs_path):
            ext = os.path.splitext(abs_path)[1].lower()
            mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                    "gif": "image/gif", "svg": "image/svg+xml"}.get(ext.lstrip("."), "image/png")
            with open(abs_path, "rb") as img_f:
                b64 = base64.b64encode(img_f.read()).decode()
            return f'src="data:{mime};base64,{b64}"'
        return match.group(0)
    return re.sub(r'src="([^"]+)"', replacer, html)

html_body = embed_images(html_body, BASE_DIR)

css = """
@page {
    size: letter;
    margin: 1in;
}
* { box-sizing: border-box; }
body {
    font-family: 'Times New Roman', 'Georgia', serif;
    font-size: 11pt;
    line-height: 1.55;
    color: #1a1a1a;
    max-width: 7.5in;
    margin: 0 auto;
    padding: 0;
}
h1 {
    font-size: 17pt;
    text-align: center;
    margin: 0 0 4pt 0;
    padding: 0;
}
h2 {
    font-size: 13pt;
    border-bottom: 1.5px solid #333;
    padding-bottom: 3pt;
    margin-top: 16pt;
    margin-bottom: 6pt;
    page-break-after: avoid;
}
h3 {
    font-size: 11.5pt;
    margin-top: 10pt;
    margin-bottom: 4pt;
    page-break-after: avoid;
}
p {
    text-align: justify;
    margin: 4pt 0;
    orphans: 3; widows: 3;
}
img {
    max-width: 85%;
    height: auto;
    display: block;
    margin: 6pt auto;
    page-break-inside: avoid;
}
table {
    width: 100%;
    border-collapse: collapse;
    margin: 8pt 0;
    font-size: 9.5pt;
    page-break-inside: avoid;
}
th {
    background: #2c3e50;
    color: #fff;
    padding: 5pt 6pt;
    text-align: center;
    font-weight: bold;
    border: 1px solid #2c3e50;
}
td {
    padding: 4pt 6pt;
    border: 1px solid #ddd;
    text-align: center;
    vertical-align: top;
}
tr:nth-child(even) td { background: #f8f9fa; }
td img {
    max-width: 220px;
    margin: 4pt auto;
}
em { color: #444; font-size: 10pt; }
code {
    background: #f0f0f0;
    padding: 1px 4px;
    border-radius: 3px;
    font-size: 9.5pt;
    font-family: 'Courier New', monospace;
}
pre {
    background: #f4f4f4;
    padding: 8pt;
    border-radius: 4px;
    font-size: 9pt;
    overflow-x: auto;
    page-break-inside: avoid;
}
pre code { background: none; padding: 0; }
hr {
    border: none;
    border-top: 1px solid #ccc;
    margin: 12pt 0;
}
ul, ol { margin: 4pt 0; padding-left: 22pt; }
li { margin: 2pt 0; }
strong { color: #111; }
a { color: #2980b9; text-decoration: none; }

@media print {
    body { margin: 0; max-width: 100%; }
    img { max-width: 80%; }
    h2, h3 { page-break-after: avoid; }
    table, pre, img { page-break-inside: avoid; }
    td img { max-width: 200px; }
}
"""

full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Final Project Report - Training and Deployment Safety in RL</title>
<style>{css}</style>
</head>
<body>
{html_body}
</body>
</html>"""

output_path = os.path.join(BASE_DIR, "Final_Project_Report.html")
with open(output_path, "w") as f:
    f.write(full_html)
print(f"HTML generated: {output_path}")
print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")
print("\\nOpen this file in Chrome/Firefox, then Ctrl+P → Save as PDF.")
print("Set margins to Default/Normal and enable Background graphics.")
