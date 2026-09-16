"""Figure 1: render the paper's TikZ architecture schematic in the shared palette.

Requires pdflatex with TikZ and pdftoppm (Poppler). All inputs are bundled.
"""
from pathlib import Path
import shutil
import subprocess
import tempfile

from common import DATA, PDF, PNG, arm_color


def main():
    for executable in ("pdflatex", "pdftoppm"):
        if shutil.which(executable) is None:
            raise SystemExit(f"Figure 1 requires {executable}; see plotting_scripts/README.md")
    palette = []
    for name, arm in {
        "archVan": "vanilla", "archK1": "operator_1", "archK2": "loop_2",
        "archGrowK2": "loop_grow", "archDep": "untied_2", "archGrowDep": "untied_grow",
    }.items():
        palette.append(rf"\definecolor{{{name}}}{{HTML}}{{{arm_color(arm).lstrip('#')}}}")
        palette.append(rf"\colorlet{{{name}Fill}}{{{name}!40!white}}")
    for name, value in {
        "archFill": "FFFFFF", "archFillStroke": "555555", "archInk": "111111",
        "archOutline": "555555", "archWire": "57606A", "archMuted": "8B949E",
    }.items():
        palette.append(rf"\definecolor{{{name}}}{{HTML}}{{{value}}}")
    definitions, separator, picture = (DATA / "architecture.tex").read_text().partition(r"\begin{tikzpicture}")
    if not separator:
        raise ValueError("architecture.tex must contain a tikzpicture")
    document = r"""\documentclass{article}
\usepackage{tikz}
\usetikzlibrary{positioning,arrows.meta,calc,decorations.pathreplacing}
\newcommand{\Kone}{Operator-1}
\newcommand{\Ktwo}{Loop-2}
\newcommand{\Dep}{Untied-2}
\newcommand{\KtwoGrow}{Loop-Grow}
\newcommand{\DepGrow}{Untied-Grow}
""" + "\n".join(palette) + "\n" + definitions + r"""
\newsavebox{\architecturebox}
\begin{document}
\setbox\architecturebox=\hbox{\input{architecture.tex}}
\pdfpagewidth=\dimexpr\wd\architecturebox+12pt\relax
\pdfpageheight=\dimexpr\ht\architecturebox+\dp\architecturebox+12pt\relax
\hoffset=-1in
\voffset=-1in
\shipout\vbox{\kern6pt\hbox{\kern6pt\box\architecturebox\kern6pt}\kern6pt}
\end{document}
"""
    PDF.mkdir(parents=True, exist_ok=True)
    PNG.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="loop-figure1-") as directory:
        work = Path(directory)
        (work / "figure1.tex").write_text(document)
        (work / "architecture.tex").write_text(separator + picture)
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "figure1.tex"],
            cwd=work, capture_output=True, text=True,
        )
        if result.returncode:
            raise RuntimeError(f"Architecture compilation failed:\n{result.stdout}\n{result.stderr}")
        shutil.copyfile(work / "figure1.pdf", PDF / "figure1.pdf")
    subprocess.run(
        ["pdftoppm", "-png", "-singlefile", "-r", "300", str(PDF / "figure1.pdf"), str(PNG / "figure1")],
        check=True,
    )
    print(PDF / "figure1.pdf")
    print(PNG / "figure1.png")


if __name__ == "__main__":
    main()
