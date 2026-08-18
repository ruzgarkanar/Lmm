#!/usr/bin/env python3
"""Fetch the field-trial corpus: real documents, off the public internet.

WHY A SCRIPT AND NOT A FOLDER. None of these documents live in this
repository. Some are public domain, some are CC BY, one is OGL — all of them
are redistributable, and none of them are ours to *vendor*. What is version
controlled is this file: the URLs, the licence under which each one is
published, and the reason it is in the set. Run it and you get the same corpus
back; delete the folder and the repository is unchanged.

    python3 benchmarks/field/fetch_documents.py [target-dir]

Default target is `benchmarks/field/documents/`, which .gitignore excludes.

LICENCE RULE, ABSOLUTE. Only public-domain or openly licensed material is
listed here. If a document's licence could not be established, it is not in
this file — not "probably fine", not "just for testing". Every entry below
names its licence and its publisher.

Needs: `requests`-free (stdlib urllib only) except for ONE entry — the
image-only PDF, which is assembled locally from public-domain page scans and
therefore needs Pillow. That entry is skipped with a message if Pillow is
absent.
"""
import io
import os
import ssl
import sys
import urllib.error
import urllib.request

# A browser UA. Several public archives return a challenge page to the default
# python-urllib string; sending a normal one is not evasion, it is asking for
# the same bytes a browser would be handed.
_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

# name, url, licence, why it is in the set
DOCUMENTS = [
    (
        "nist_sp800-63b.pdf",
        "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-63b.pdf",
        "Public domain (work of the U.S. Government, 17 U.S.C. §105) — NIST",
        "long technical PDF with a real text layer, numbered sections, "
        "requirement tables and footnotes. The format LMM claims to be best at.",
    ),
    (
        "naca_report_scan.pdf",
        "https://ntrs.nasa.gov/api/citations/19930091001/downloads/19930091001.pdf",
        "Public domain (work of the U.S. Government) — NASA NTRS",
        "NACA Report 1270 (1956), digitised decades later: a scan carrying a "
        "POOR OCR text layer. Not the clean case and not the empty case — the "
        "common one.",
    ),
    (
        "us_constitution_scan.pdf",
        # Assembled locally; see _build_image_pdf below.
        "https://www.archives.gov/files/founding-docs/downloads/Constitution_Pg%dof4_AC.jpg",
        "Public domain — U.S. National Archives (NARA), no permission required",
        "an IMAGE-ONLY PDF, i.e. what every office scanner produces. It is in "
        "the set precisely because LMM has no OCR: this is the proof, not an "
        "accident.",
    ),
    (
        "mh370_debris_analyses.docx",
        "https://d28rz98at9flks.cloudfront.net/110564/Rec2017_011.docx",
        "CC BY 4.0 — Geoscience Australia (Record 2017/11, eCat 110564)",
        "a 14 MB real-world DOCX: a technical record with 15 tables, a long "
        "front matter section and figure captions. Not a template.",
    ),
    (
        "us_state_population.xlsx",
        "https://www2.census.gov/programs-surveys/popest/tables/2020-2023/"
        "state/totals/NST-EST2023-POP.xlsx",
        "Public domain (work of the U.S. Government) — U.S. Census Bureau",
        "a published statistical table as a real spreadsheet: merged title "
        "rows, footnote rows, and a leading-dot label column.",
    ),
    (
        "uk_cpi_timeseries.csv",
        "https://www.ons.gov.uk/generator?format=csv&uri=/economy/"
        "inflationandpriceindices/timeseries/d7g7/mm23",
        "Open Government Licence v3.0 — UK Office for National Statistics",
        "a machine-generated CSV time series: a preamble block above the "
        "header row, which is what open-data CSV actually looks like.",
    ),
    (
        "rfc9110.txt",
        "https://www.rfc-editor.org/rfc/rfc9110.txt",
        "IETF Trust / BSD-style (RFC 9110, freely distributable)",
        "a 500 KB plain-text specification: fixed-width layout, page breaks, "
        "an ASCII table of contents. The oldest document format still in use.",
    ),
    (
        "k8s_nodes.md",
        "https://raw.githubusercontent.com/kubernetes/website/main/content/en/"
        "docs/concepts/architecture/nodes.md",
        "CC BY 4.0 — The Kubernetes Authors (kubernetes/website)",
        "Markdown with YAML front matter, fenced code blocks and inline "
        "shortcodes — prose interleaved with syntax.",
    ),
    (
        "rfc9110.html",
        "https://www.rfc-editor.org/rfc/rfc9110.html",
        "IETF Trust / BSD-style (RFC 9110, freely distributable)",
        "HTML, which LMM does NOT support. Deliberately the SAME document as "
        "rfc9110.txt above, so the only variable is the container: the answer "
        "to 'what does a user who hands it a web page experience?' cannot be "
        "confounded by the content.",
    ),
    (
        "eca_sr14_2023_el.pdf",
        "https://www.eca.europa.eu/ECAPublications/SR-2023-14/SR-2023-14_EL.pdf",
        "CC BY 4.0 — European Court of Auditors, Special Report 14/2023 (NDICI-Global Europe)",
        "GREEK. Non-Latin alphabet, in a professionally typeset report PDF.",
    ),
    (
        "eca_sr14_2023_bg.pdf",
        "https://www.eca.europa.eu/ECAPublications/SR-2023-14/SR-2023-14_BG.pdf",
        "CC BY 4.0 — European Court of Auditors, Special Report 14/2023 (NDICI-Global Europe)",
        "BULGARIAN (Cyrillic). Line for line the SAME report as the Greek one "
        "— so a difference between the two is a script/language difference "
        "and nothing else.",
    ),
    (
        "eca_sr14_2023_en.pdf",
        "https://www.eca.europa.eu/ECAPublications/SR-2023-14/SR-2023-14_EN.pdf",
        "CC BY 4.0 — European Court of Auditors, Special Report 14/2023 (NDICI-Global Europe)",
        "the English original of the same report: the control arm for the "
        "Greek and Bulgarian runs.",
    ),
]


def _get(url):
    request = urllib.request.Request(url, headers={"User-Agent": _UA})
    context = ssl.create_default_context()
    with urllib.request.urlopen(request, timeout=180, context=context) as r:
        return r.read()


def _build_image_pdf(target, url_pattern):
    """The scanned case, assembled honestly.

    Public archives that publish page scans overwhelmingly publish them as
    images, and the PDFs they do publish have had OCR run over them. To get an
    HONEST image-only PDF — the thing a user's scanner hands them — we take
    NARA's public-domain page scans and wrap them, adding no text layer. The
    result is exactly what a scan-to-PDF button produces, and it contains not
    one extractable character. That is the point of the entry.
    """
    try:
        from PIL import Image
    except ImportError:
        print("  skipped: needs Pillow (pip install pillow) to wrap the scans")
        return False
    pages = []
    for n in range(1, 5):
        raw = _get(url_pattern % n)
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        # Downscale: NARA's masters are enormous and the point of the file is
        # its structure, not its resolution.
        image.thumbnail((1700, 2200))
        pages.append(image)
    pages[0].save(target, save_all=True, append_images=pages[1:])
    return True


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    target_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "documents")
    os.makedirs(target_dir, exist_ok=True)

    for name, url, licence, why in DOCUMENTS:
        target = os.path.join(target_dir, name)
        print(f"\n{name}\n  licence: {licence}\n  why: {why}")
        if os.path.exists(target) and os.path.getsize(target) > 0:
            print(f"  already here ({os.path.getsize(target):,} bytes)")
            continue
        try:
            if name == "us_constitution_scan.pdf":
                if not _build_image_pdf(target, url):
                    continue
            else:
                data = _get(url)
                with open(target, "wb") as handle:
                    handle.write(data)
            print(f"  fetched {os.path.getsize(target):,} bytes")
        except (urllib.error.URLError, OSError) as gone:
            print(f"  FAILED: {gone}")

    print(f"\n{len(DOCUMENTS)} documents → {target_dir}")
    print("None of them are committed. See benchmarks/field/REPORT.md.")


if __name__ == "__main__":
    main()
