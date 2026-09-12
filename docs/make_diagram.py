"""Builds the README's architecture diagram — one template, two themes.

GitHub serves a README image in whichever theme the reader is in, through the
`<picture>` element, which means two files. They are generated from one
description here rather than drawn twice, because two hand-kept SVGs drift and
the drift is invisible until someone reads the repository in the theme nobody
checked.

What the picture has to say, in order of importance:

    THE GATE APPEARS TWICE.  Same shape, once on the way in and once on the
    way out. It is the one rule the architecture rests on, and a diagram that
    draws it once has drawn a retrieval pipeline instead.

    THE TABLE PATH SKIPS THE ENGINE.  A spreadsheet states its own structure,
    so it reaches the graph with no model call at all. That is why a sheet
    loads in milliseconds, and it is invisible unless the arrow is drawn.

    THE ENGINE IS SMALL.  Deliberately drawn smaller than the memory: it does
    not hold the knowledge, and the whole argument is downstream of that.

    RETRIEVAL HAS TWO CHANNELS AND NEITHER IS THE ENGINE.  Words and meaning
    are read side by side and fused by rank; the vectors ship with the
    library. Drawing only the word channel would draw the system as it was
    before 0.6.

    AN ANSWER CAN BE DROPPED.  The exit has two outcomes, and the second one
    is not an error state — it is the feature.

Usage:
    python3 docs/make_diagram.py
"""
import os

LIGHT = {"ink": "#14181F", "slate": "#5A6472", "rule": "#C9CFD8",
         "teal": "#0E7C86", "bronze": "#A6712E", "wash": "#EEF3F4",
         "faint": "#F5F6F8", "drop": "#8A93A0"}

DARK = {"ink": "#E6E9EF", "slate": "#95A0B0", "rule": "#39424F",
        "teal": "#3BC4C9", "bronze": "#D6A050", "wash": "#12262A",
        "faint": "#171E27", "drop": "#7D8794"}

FONT = ("ui-sans-serif,-apple-system,BlinkMacSystemFont,'Segoe UI',"
        "Helvetica,Arial,sans-serif")
MONO = "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"

TEMPLATE = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 980 470"
     width="980" height="470" role="img"
     aria-label="LMM architecture: documents, tables and messages pass an entry gate into a graph memory holding facts, evidence sentences and a bundled meaning channel; a question retrieves from that memory by words and by meaning, a small engine phrases the answer, and an exit gate either releases it or drops it.">
  <defs>
    <marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7"
            markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="{slate}"/>
    </marker>
    <marker id="t" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7"
            markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="{teal}"/>
    </marker>
    <marker id="d" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7"
            markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="{drop}"/>
    </marker>
  </defs>

  <style>
    .l  {{ font-family: {font}; font-size: 14px; fill: {ink}; }}
    .s  {{ font-family: {font}; font-size: 12.5px; fill: {slate}; }}
    .k  {{ font-family: {mono}; font-size: 11px; fill: {slate};
           letter-spacing: .09em; }}
    .g  {{ font-family: {mono}; font-size: 12px; fill: {teal};
           letter-spacing: .09em; font-weight: 600; }}
    .h  {{ font-family: {font}; font-size: 15px; fill: {ink};
           font-weight: 600; }}
    .box {{ fill: {faint}; stroke: {rule}; stroke-width: 1; }}
    .mem {{ fill: {wash}; stroke: {teal}; stroke-width: 1.5; }}
    .eng {{ fill: none; stroke: {bronze}; stroke-width: 1.5; }}
    .ln  {{ stroke: {slate}; stroke-width: 1.2; fill: none; }}
    .lt  {{ stroke: {teal}; stroke-width: 1.6; fill: none; }}
    .ld  {{ stroke: {drop}; stroke-width: 1.2; fill: none;
            stroke-dasharray: 4 3; }}
  </style>

  <!-- ingestion label -->
  <text x="24" y="30" class="k">INGESTION</text>

  <!-- inputs -->
  <rect x="24" y="48" width="132" height="38" rx="2" class="box"/>
  <text x="38" y="66" class="l">prose document</text>
  <text x="38" y="80" class="s">pdf · docx · md · txt</text>

  <rect x="24" y="100" width="132" height="38" rx="2" class="box"/>
  <text x="38" y="118" class="l">spreadsheet</text>
  <text x="38" y="132" class="s">xlsx · csv</text>

  <rect x="24" y="152" width="132" height="38" rx="2" class="box"/>
  <text x="38" y="170" class="l">message</text>
  <text x="38" y="184" class="s">what someone tells it</text>

  <!-- extraction (engine) — prose and message only -->
  <rect x="212" y="48" width="104" height="38" rx="2" class="eng"/>
  <text x="226" y="66" class="l">extract</text>
  <text x="226" y="80" class="s">engine</text>

  <path d="M156,67 L206,67" class="ln" marker-end="url(#a)"/>
  <path d="M156,171 C186,171 186,86 206,79" class="ln" marker-end="url(#a)"/>

  <!-- the table path skips the engine entirely -->
  <path d="M156,119 L372,119" class="lt" marker-end="url(#t)"/>
  <text x="196" y="112" class="g">NO MODEL CALL</text>

  <path d="M316,67 L372,67" class="ln" marker-end="url(#a)"/>

  <!-- entry gate -->
  <rect x="378" y="34" width="26" height="170" rx="2"
        fill="none" stroke="{teal}" stroke-width="1.8"/>
  <line x1="378" y1="72" x2="404" y2="72" stroke="{teal}" stroke-width="1.8"/>
  <line x1="378" y1="110" x2="404" y2="110" stroke="{teal}" stroke-width="1.8"/>
  <line x1="378" y1="148" x2="404" y2="148" stroke="{teal}" stroke-width="1.8"/>
  <text x="391" y="222" class="g" text-anchor="middle">GATE</text>
  <text x="391" y="238" class="s" text-anchor="middle">source · conflict</text>

  <path d="M404,119 L452,119" class="lt" marker-end="url(#t)"/>

  <!-- memory -->
  <rect x="458" y="34" width="214" height="170" rx="3" class="mem"/>
  <text x="474" y="58" class="h">memory</text>
  <text x="474" y="80" class="l">graph</text>
  <text x="474" y="96" class="s">facts · source · trust · time</text>
  <text x="474" y="116" class="l">links between facts</text>
  <text x="474" y="132" class="s">cause · then · contradicts</text>
  <text x="474" y="150" class="l">evidence index</text>
  <text x="474" y="164" class="s">sentences verbatim · words</text>
  <text x="474" y="180" class="l">meaning channel</text>
  <text x="474" y="194" class="s">vectors, bundled · no model call</text>

  <!-- answering -->
  <text x="24" y="292" class="k">ANSWERING</text>

  <rect x="24" y="310" width="132" height="38" rx="2" class="box"/>
  <text x="38" y="328" class="l">question</text>
  <text x="38" y="342" class="s">in any language</text>

  <path d="M156,329 H502 V212" class="ln" marker-end="url(#a)"/>
  <text x="196" y="322" class="k">RETRIEVE</text>
  <text x="196" y="344" class="s">graph lookup · words + meaning, fused by rank, reordered — no model call</text>

  <!-- memory feeds the engine -->
  <path d="M565,204 L565,300" class="lt" marker-end="url(#t)"/>
  <text x="576" y="248" class="g">ONLY THESE</text>
  <text x="576" y="264" class="g">RECORDS</text>

  <rect x="504" y="310" width="122" height="38" rx="2" class="eng"/>
  <text x="518" y="328" class="l">phrase it</text>
  <text x="518" y="342" class="s">engine · small</text>

  <path d="M626,329 L688,329" class="ln" marker-end="url(#a)"/>

  <!-- exit gate -->
  <rect x="694" y="296" width="26" height="66" rx="2"
        fill="none" stroke="{teal}" stroke-width="1.8"/>
  <line x1="694" y1="318" x2="720" y2="318" stroke="{teal}" stroke-width="1.8"/>
  <line x1="694" y1="340" x2="720" y2="340" stroke="{teal}" stroke-width="1.8"/>
  <text x="707" y="382" class="g" text-anchor="middle">GATE</text>
  <text x="707" y="398" class="s" text-anchor="middle">re-read · supported?</text>

  <!-- outcomes -->
  <path d="M720,318 L806,300" class="lt" marker-end="url(#t)"/>
  <rect x="812" y="278" width="144" height="38" rx="2" class="mem"/>
  <text x="826" y="296" class="l">answer</text>
  <text x="826" y="310" class="s">with its sources</text>

  <path d="M720,340 L806,362" class="ld" marker-end="url(#d)"/>
  <rect x="812" y="342" width="144" height="38" rx="2"
        fill="none" stroke="{drop}" stroke-width="1"
        stroke-dasharray="4 3"/>
  <text x="826" y="360" class="l">dropped</text>
  <text x="826" y="374" class="s">not repaired</text>

  <!-- the rule -->
  <line x1="24" y1="432" x2="956" y2="432" stroke="{rule}" stroke-width="1"/>
  <text x="24" y="456" class="s">The engine cannot write records, and cannot speak unsupported facts. Every other property is downstream of that sentence.</text>
</svg>
"""


def build(colours, path):
    svg = TEMPLATE.format(font=FONT, mono=MONO, **colours)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(svg)
    return path


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    for colours, name in ((LIGHT, "architecture-light.svg"),
                          (DARK, "architecture-dark.svg")):
        print(build(colours, os.path.join(here, name)))
