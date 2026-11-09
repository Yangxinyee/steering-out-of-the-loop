#!/usr/bin/env python3
# Vendored from the authors' paper-figures toolkit (figure helpers used by paper/figures/make_figures.py).
# Released here under the MIT License, as the rest of this repository (see LICENSE).
"""figkit: single-source publication diagrams (vector PDF + SVG + native Draw.io).

Import it from a figure script:

    import sys; sys.path.insert(0, '<skill>/scripts')
    from figkit import *
    use_theme('B')                      # 'A' clinical editorial, 'B' slate + one accent, 'C' mechanism-first
    d = Figure('fig03_modes', 334, 'Authority modes', out='figures')
    d.box(16, 12, 300, 80, T.panel, None, T.r_outer)
    d.text(30, 24, 270, 'Copilot', T.title, True)
    ...
    d.save()                            # writes fig03_modes.pdf / .svg / .drawio
    render('figures/fig03_modes.pdf')   # 220 dpi colour PNG + greyscale PNG

Coordinates: 1000 units across the page; the PDF page is `width_pt` wide
(516 pt = IEEE double-column width; 252 pt = one IEEE column; 510 pt = Nature
180 mm). 1 unit = width_pt/1000 points, so with 516 pt a 15.5-unit label prints
at 8.0 pt. Text that overflows its box is recorded in ERRORS; call
`check_errors()` at the end of a build to fail loudly.

Needs: reportlab. Optional: ghostscript (`gs`) or PyMuPDF for PNG rendering, Pillow for contact sheets.
"""
from pathlib import Path
import json, math, html, os, shutil, subprocess
import xml.etree.ElementTree as E
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor

ALL, ERRORS = [], []

# ---- fonts -------------------------------------------------------------------
# Each candidate is (regular path, face, bold path, face). The first that exists is used.
FONT_CANDIDATES = {
    'helvetica-neue': [('/System/Library/Fonts/HelveticaNeue.ttc', 0, '/System/Library/Fonts/HelveticaNeue.ttc', 1)],
    'arial': [('/System/Library/Fonts/Supplemental/Arial.ttf', 0, '/System/Library/Fonts/Supplemental/Arial Bold.ttf', 0),
              ('/Library/Fonts/Arial.ttf', 0, '/Library/Fonts/Arial Bold.ttf', 0),
              ('C:/Windows/Fonts/arial.ttf', 0, 'C:/Windows/Fonts/arialbd.ttf', 0),
              ('/usr/share/fonts/truetype/msttcorefonts/Arial.ttf', 0, '/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf', 0)],
    'liberation': [('/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf', 0, '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf', 0),
                   ('/usr/share/fonts/liberation-sans/LiberationSans-Regular.ttf', 0, '/usr/share/fonts/liberation-sans/LiberationSans-Bold.ttf', 0)],
    'dejavu': [('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 0, '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 0)],
}
SANS, SANS_BOLD, FONT_FAMILY = 'FigSans', 'FigSans-Bold', 'Helvetica Neue, Helvetica, Arial, sans-serif'

def use_font(prefer=('helvetica-neue', 'arial', 'liberation', 'dejavu')):
    """Register the first available family; Helvetica/Arial-class faces match IEEE and Nature guidance."""
    global FONT_FAMILY
    for fam in prefer:
        for reg, rf, bold, bf in FONT_CANDIDATES.get(fam, []):
            if os.path.exists(reg) and os.path.exists(bold):
                pdfmetrics.registerFont(TTFont(SANS, reg, subfontIndex=rf))
                pdfmetrics.registerFont(TTFont(SANS_BOLD, bold, subfontIndex=bf))
                FONT_FAMILY = {'helvetica-neue': 'Helvetica Neue, Helvetica, Arial, sans-serif', 'arial': 'Arial, Helvetica, sans-serif',
                               'liberation': 'Liberation Sans, Arial, sans-serif', 'dejavu': 'DejaVu Sans, sans-serif'}[fam]
                return fam
    raise RuntimeError('No usable sans-serif TTF found; add its path to FONT_CANDIDATES')

# ---- themes ------------------------------------------------------------------
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from themes import Theme, THEMES, WHITE, luminance
T = THEMES['B']

def on(fill):
    """Text colour for a fill: white on dark fills, the theme ink otherwise (keeps contrast at 4.5:1 or better)."""
    return WHITE if (1.05) / (luminance(fill) + .05) >= 4.5 else T.ink


WIDTH = 516.0

def tex_to_bp(tex_pt):
    """LaTeX reports \\textwidth in TeX points (1/72.27 in); PDF tools use big points (1/72 in)."""
    return tex_pt * 72 / 72.27

def use_theme(key_or_theme, width_pt=516, font=None):
    """Select a theme, register its font, and size it for the printed width (in PDF points, bp).

    Theme values are tuned for 516 bp (IEEE double column). For another width every size in units is
    rescaled -- type, line widths, corner radii and glyph strokes -- so the printed result is the same:
    body near 8.5 pt, nothing below 8 pt, keylines 0.5 pt, connectors 0.75 pt. Figures created afterwards
    take this width unless they pass their own. font: optional family key ('arial', 'helvetica-neue', ...),
    e.g. 'arial' to match matplotlib plots exactly."""
    global T, WIDTH
    import copy
    base = THEMES[key_or_theme] if isinstance(key_or_theme, str) else key_or_theme
    T = copy.deepcopy(base)
    f = 516 / width_pt
    for k in ('body', 'item', 'small', 'title', 'zone', 'lw_box', 'lw_conn', 'lw_loop', 'r_outer', 'r_inner'):
        setattr(T, k, round(getattr(base, k) * f, 3))
    T.scale, WIDTH = f, float(width_pt)
    if font: T.font = (font,) + tuple(x for x in base.font if x != font)
    use_font(T.font)
    return T

# ---- the figure ----------------------------------------------------------------
class Figure:
    def __init__(self, stem, h, title, out='.', width_pt=None):
        width_pt = WIDTH if width_pt is None else width_pt   # default: the width given to use_theme()
        self.stem, self.h, self.out, self.k = stem, h, Path(out), width_pt / 1000
        self.out.mkdir(parents=True, exist_ok=True)
        self.svg, self.texts, self.n = [], [], 1
        # Start in the embedded figure font so the PDF carries no reference to a non-embedded base font.
        self.c = canvas.Canvas(str(self.out / (stem + '.pdf')), pagesize=(width_pt, h * self.k), pageCompression=1, initialFontName=SANS)
        self.c.setTitle(title); self.c.scale(self.k, self.k)
        self.xml = E.Element('mxfile', host='app.diagrams.net', version='24.7.17')
        dg = E.SubElement(self.xml, 'diagram', id=stem, name=title)
        md = E.SubElement(dg, 'mxGraphModel', page='1', pageWidth='1000', pageHeight=str(h), grid='0')
        self.rx = E.SubElement(md, 'root'); E.SubElement(self.rx, 'mxCell', id='0'); E.SubElement(self.rx, 'mxCell', id='1', parent='0')
        self.box(0, 0, 1000, h, WHITE, None, 0)   # explicit white page

    def pt(self, units): return units * self.k
    def cell(self, x, y, w, h, style, value=''):
        self.n += 1
        el = E.SubElement(self.rx, 'mxCell', id=str(self.n), value=value, style=style, vertex='1', parent='1')
        E.SubElement(el, 'mxGeometry', x=f'{x:.2f}', y=f'{y:.2f}', width=f'{w:.2f}', height=f'{h:.2f}', attrib={'as': 'geometry'})

    def box(self, x, y, w, h, fill=WHITE, stroke='theme', r=None, lw=None, dash=False):
        """Rectangle with corner radius r. stroke='theme' uses the theme's keyline colour; None draws no outline."""
        stroke = T.line if stroke == 'theme' else stroke
        r = T.r_inner if r is None else r; lw = T.lw_box if lw is None else lw
        c = self.c; c.setLineWidth(lw); c.setDash([4, 3] if dash else [])
        if fill: c.setFillColor(HexColor(fill))
        if stroke: c.setStrokeColor(HexColor(stroke))
        c.roundRect(x, self.h - y - h, w, h, r, fill=int(bool(fill)), stroke=int(bool(stroke)))
        self.svg.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" rx="{r}" fill="{fill or "none"}" stroke="{stroke or "none"}" stroke-width="{lw}"' + (' stroke-dasharray="4 3"' if dash else '') + '/>')
        self.cell(x, y, w, h, f'rounded={int(r > 0)};arcSize=8;fillColor={fill or "none"};strokeColor={stroke or "none"};strokeWidth={lw};dashed={int(dash)};')

    def text(self, x, y, w, t, size=None, bold=False, color=None, align='left'):
        """Lines split on \\n. (x, y) is the top of the first line box; line height is 1.24 x size."""
        size = T.body if size is None else size; color = T.ink if color is None else color
        font = SANS_BOLD if bold else SANS; lines = t.split('\n'); lh = size * 1.24
        for i, ln in enumerate(lines):
            width = pdfmetrics.stringWidth(ln, font, size)
            if width > w + .5: ERRORS.append(f'{self.stem}: text too wide ({width:.1f}>{w:.1f}): {ln}')
            xx = x if align == 'left' else x + (w - width) / 2 if align == 'center' else x + w - width
            baseline = y + size * .83 + i * lh
            if xx < -0.5 or xx + width > 1000.5 or baseline > self.h: ERRORS.append(f'{self.stem}: text outside page: {ln}')
            self.c.setFont(font, size); self.c.setFillColor(HexColor(color)); self.c.drawString(xx, self.h - baseline, ln)
            self.svg.append(f'<text x="{xx:.2f}" y="{baseline:.2f}" font-family="{FONT_FAMILY}" font-size="{size}" font-weight="{700 if bold else 400}" fill="{color}">{html.escape(ln)}</text>')
            self.texts.append(dict(text=ln, x=xx, y=y + i * lh, width=width, size=size, printed_pt=size * self.k, color=color))
        self.cell(x, y, w, len(lines) * lh, f'text;html=0;whiteSpace=wrap;overflow=hidden;align={align};verticalAlign=top;spacing=0;fontFamily={FONT_FAMILY.split(",")[0]};fontSize={size};fontColor={color};fontStyle={int(bold)};strokeColor=none;fillColor=none;', t)

    def ctext(self, x, y, w, h, t, size=None, bold=False, color=None, align='center'):
        """Text block centred vertically in the box (x, y, w, h)."""
        size = T.body if size is None else size
        self.text(x, y + (h - th(t, size)) / 2 + size * .06, w, t, size, bold, color, align)

    def vtext(self, x, ymid, t, size=None, color=None, bold=False):
        """One line rotated 90 degrees (reads bottom to top), centred on ymid; glyphs occupy x..x+size."""
        size = T.small if size is None else size; color = T.muted if color is None else color
        font = SANS_BOLD if bold else SANS; w = pdfmetrics.stringWidth(t, font, size); c = self.c; c.saveState()
        c.setFillColor(HexColor(color)); c.setFont(font, size)
        c.translate(x + size * .76, self.h - (ymid + w / 2)); c.rotate(90); c.drawString(0, 0, t); c.restoreState()
        self.svg.append(f'<text transform="translate({x + size * .76:.1f},{ymid + w / 2:.1f}) rotate(-90)" font-family="{FONT_FAMILY}" font-size="{size}" font-weight="{700 if bold else 400}" fill="{color}">{html.escape(t)}</text>')
        self.texts.append(dict(text=t, x=x, y=ymid - w / 2, width=w, size=size, printed_pt=size * self.k, color=color))
        self.cell(x, ymid - w / 2, size * 1.24, w, f'text;html=0;horizontal=0;align=center;verticalAlign=middle;spacing=0;fontSize={size};fontColor={color};fontStyle={int(bold)};strokeColor=none;fillColor=none;', t)

    def line(self, points, color=None, lw=None, dash=False, arrow=False):
        color = T.ramp[-1] if color is None else color; lw = T.lw_conn if lw is None else lw
        c = self.c; c.setStrokeColor(HexColor(color)); c.setLineWidth(lw); c.setDash([5, 4] if dash else [])
        p = c.beginPath(); p.moveTo(points[0][0], self.h - points[0][1])
        for x, y in points[1:]: p.lineTo(x, self.h - y)
        c.drawPath(p)
        self.svg.append('<polyline points="' + ' '.join(f'{x:.2f},{y:.2f}' for x, y in points) + f'" fill="none" stroke="{color}" stroke-width="{lw}"' + (' stroke-dasharray="5 4"' if dash else '') + '/>')
        head = 'endArrow=block;endFill=1;endSize=6;' if arrow else 'endArrow=none;'
        self.n += 1; el = E.SubElement(self.rx, 'mxCell', id=str(self.n), style=f'edgeStyle=none;rounded=0;{head}strokeColor={color};strokeWidth={lw};dashed={int(dash)};', edge='1', parent='1')
        g = E.SubElement(el, 'mxGeometry', relative='1', attrib={'as': 'geometry'})
        for pt_, label in [(points[0], 'sourcePoint'), (points[-1], 'targetPoint')]: E.SubElement(g, 'mxPoint', x=f'{pt_[0]:.2f}', y=f'{pt_[1]:.2f}', attrib={'as': label})
        if len(points) > 2:
            a = E.SubElement(g, 'Array', attrib={'as': 'points'})
            for x, y in points[1:-1]: E.SubElement(a, 'mxPoint', x=f'{x:.2f}', y=f'{y:.2f}')
        if arrow:
            # Filled head on the direction of the last segment; scales a little with the stroke.
            x, y = points[-1]; x0, y0 = points[-2]; ang = math.atan2(y - y0, x - x0); L, W = 6 * T.scale + 1.4 * lw, 2.6 * T.scale + .7 * lw
            self.poly([(x, y), (x - L * math.cos(ang) + W * math.sin(ang), y - L * math.sin(ang) - W * math.cos(ang)),
                       (x - L * math.cos(ang) - W * math.sin(ang), y - L * math.sin(ang) + W * math.cos(ang))], color, color, .6 * T.scale, native=False)

    def poly(self, pts, fill, stroke=None, lw=1, native=True):
        c = self.c; p = c.beginPath(); p.moveTo(pts[0][0], self.h - pts[0][1])
        for x, y in pts[1:]: p.lineTo(x, self.h - y)
        p.close(); c.setDash([]); c.setFillColor(HexColor(fill))
        if stroke: c.setStrokeColor(HexColor(stroke)); c.setLineWidth(lw)
        c.drawPath(p, fill=1, stroke=1 if stroke else 0)
        self.svg.append('<polygon points="' + ' '.join(f'{x:.2f},{y:.2f}' for x, y in pts) + f'" fill="{fill}" stroke="{stroke or "none"}" stroke-width="{lw}"/>')
        if native:
            x0, y0 = min(x for x, _ in pts), min(y for _, y in pts); w, h = max(x for x, _ in pts) - x0, max(y for _, y in pts) - y0
            coords = json.dumps([[round((x - x0) / max(w, 1e-6), 4), round((y - y0) / max(h, 1e-6), 4)] for x, y in pts], separators=(',', ':'))
            self.cell(x0, y0, w, h, f'shape=mxgraph.basic.polygon;polyCoords={coords};polyline=0;fillColor={fill};strokeColor={stroke or "none"};strokeWidth={lw};')

    def circle(self, x, y, r, stroke=None, fill=WHITE, lw=1.5):
        stroke = T.ink if stroke is None else stroke
        self.c.setLineWidth(lw); self.c.setDash([]); self.c.setStrokeColor(HexColor(stroke)); self.c.setFillColor(HexColor(fill)); self.c.circle(x, self.h - y, r, fill=1, stroke=1)
        self.svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{lw}"/>')
        self.cell(x - r, y - r, 2 * r, 2 * r, f'ellipse;fillColor={fill};strokeColor={stroke};strokeWidth={lw};')

    def hatch(self, x, y, w, h, color=None, step=6, lw=1.2):
        """Diagonal hatching clipped to a rectangle: a second variable that survives greyscale (PDF only)."""
        color = T.ink if color is None else color
        c = self.c; c.saveState(); p = c.beginPath(); p.rect(x, self.h - y - h, w, h); c.clipPath(p, stroke=0, fill=0)
        c.setStrokeColor(HexColor(color)); c.setLineWidth(lw); c.setDash([])
        for k in range(-int(h) - step, int(w) + int(h), step): c.line(x + k, self.h - y - h, x + k + h, self.h - y)
        c.restoreState()

    def save(self):
        self.c.save()
        (self.out / (self.stem + '.svg')).write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="{self.h}" viewBox="0 0 1000 {self.h}">\n' + '\n'.join(self.svg) + '\n</svg>')
        if hasattr(E, 'indent'): E.indent(self.xml)   # Python >= 3.9; older versions write the same XML unindented
        E.ElementTree(self.xml).write(self.out / (self.stem + '.drawio'), encoding='utf-8', xml_declaration=True)
        words = sum(1 for t in self.texts for w in t['text'].split() if any(ch.isalnum() for ch in w))
        info = dict(stem=self.stem, height_units=self.h, height_pt=round(self.h * self.k, 1), words=words,
                    min_pt=round(min((t['printed_pt'] for t in self.texts), default=0), 2), text=self.texts)
        ALL.append(info)
        return info

# ---- text helpers --------------------------------------------------------------
def tw(t, size, bold=False):
    return max(pdfmetrics.stringWidth(l, SANS_BOLD if bold else SANS, size) for l in t.split('\n'))

def wrap_words(t, width, size, bold=False):
    out = []
    for para in t.split('\n'):
        cur = ''
        for w in para.split(' '):
            cand = (cur + ' ' + w) if cur else w
            if tw(cand, size, bold) <= width: cur = cand
            else: out.append(cur); cur = w
        out.append(cur)
    return '\n'.join(out)

def wrap_items(items, width, size, bold=False, sep=' · '):
    """Pack short items into as few lines as fit, separated by a middle dot."""
    lines, cur = [], ''
    for it in items:
        cand = cur + sep + it if cur else it
        if tw(cand, size, bold) <= width: cur = cand
        else:
            if cur: lines.append(cur)
            cur = it
    lines.append(cur)
    return '\n'.join(lines)

def nlines(t): return len(t.split('\n'))
def th(t, size): return nlines(t) * size * 1.24

# ---- geometry helpers ------------------------------------------------------------
def arc(cx, cy, r, a0, a1, n=18):
    return [(cx + r * math.cos(math.radians(a0 + (a1 - a0) * i / n)), cy + r * math.sin(math.radians(a0 + (a1 - a0) * i / n))) for i in range(n + 1)]

def rounded(pts, r, n=8):
    """A polyline with its interior corners rounded to radius r (quadratic curves through each corner)."""
    out = [pts[0]]
    for (x0, y0), (x1, y1), (x2, y2) in zip(pts, pts[1:], pts[2:]):
        d1, d2 = math.hypot(x1 - x0, y1 - y0), math.hypot(x2 - x1, y2 - y1); k = min(r, d1 / 2, d2 / 2)
        a = (x1 - (x1 - x0) * k / d1, y1 - (y1 - y0) * k / d1); b = (x1 + (x2 - x1) * k / d2, y1 + (y2 - y1) * k / d2)
        out += [((1 - s) ** 2 * a[0] + 2 * (1 - s) * s * x1 + s * s * b[0], (1 - s) ** 2 * a[1] + 2 * (1 - s) * s * y1 + s * s * b[1]) for s in (i / n for i in range(n + 1))]
    out.append(pts[-1])
    return out

def arrow2(d, pts, color=None, lw=None):
    """Double-headed connector. Label it with one direction above the line and the other below, without a slash."""
    d.line(pts, color, lw, arrow=True); d.line(list(reversed(pts)), color, lw, arrow=True)

def link_labels(d, x, w, y, above, below, size=None, color=None, gap=5):
    """Labels for a horizontal double-headed link at height y, centred on the span x..x+w."""
    size = T.small if size is None else size; color = T.muted if color is None else color
    d.text(x, y - gap - size * 1.24, w, above, size, color=color, align='center')
    d.text(x, y + gap, w, below, size, color=color, align='center')

# ---- glyphs and marks -----------------------------------------------------------
def ellipse(cx, cy, rx, ry, a0, a1, n=18):
    return [(cx + rx * math.cos(math.radians(a0 + (a1 - a0) * i / n)), cy + ry * math.sin(math.radians(a0 + (a1 - a0) * i / n))) for i in range(n + 1)]

def glyph(d, kind, x, y, s, fg=WHITE, bg=None, lw=None):
    """One consistent line-icon set in the unit square at (x, y), side s. Draw on a tile (see tile())."""
    bg = T.ramp[-1] if bg is None else bg; lw = 1.5 * T.scale if lw is None else lw
    P = lambda pts: [(x + a * s, y + b * s) for a, b in pts]
    def l(pts): d.line(P(pts), fg, lw)
    def bx(a, b, w, h, r=1.2): d.box(x + a * s, y + b * s, w * s, h * s, bg, fg, r, lw)
    def o(a, b, r): d.circle(x + a * s, y + b * s, r * s, fg, bg, lw)
    def person(cx, cy, k=1.0):
        o(cx, cy, .15 * k); l([(cx - .32 * k, cy + .68 * k), (cx - .32 * k, cy + .45 * k), (cx - .18 * k, cy + .27 * k), (cx + .18 * k, cy + .27 * k), (cx + .32 * k, cy + .45 * k), (cx + .32 * k, cy + .68 * k)])
    if kind == 'patient': person(.68, .3, .85); person(.34, .26, 1.0)
    elif kind == 'person': person(.5, .22, 1.1)
    elif kind == 'clinician': person(.5, .22, 1.1); l([(.5, .62), (.5, .88)]); l([(.38, .75), (.62, .75)])
    elif kind == 'hospital':
        bx(.12, .3, .76, .66); bx(.4, .7, .2, .26, .5); l([(.5, .02), (.5, .24)]); l([(.4, .13), (.6, .13)]); l([(.24, .46), (.36, .46)]); l([(.64, .46), (.76, .46)])
    elif kind == 'eye': l(arc(.5, .767, .567, 208.1, 331.9)); l(arc(.5, .233, .567, 28.1, 151.9)); o(.5, .5, .14)
    elif kind == 'network':
        pts = [(.2, .22), (.8, .22), (.5, .8)]
        for a in pts:
            for b in pts: l([a, b])
        for a, b in pts: o(a, b, .12)
    elif kind == 'stack':
        for yy in (.08, .38, .68): bx(.12, yy, .76, .22, 1.5)
    elif kind == 'search': o(.42, .42, .3); l([(.64, .64), (.94, .94)])
    elif kind == 'api': l([(.3, .2), (.06, .5), (.3, .8)]); l([(.7, .2), (.94, .5), (.7, .8)]); l([(.58, .1), (.42, .9)])
    elif kind == 'escalate': o(.5, .5, .44); l([(.5, .78), (.5, .24)]); l([(.3, .44), (.5, .24), (.7, .44)])
    elif kind in ('log', 'doc'):
        bx(.16, .04, .68, .92); l([(.3, .3), (.7, .3)]); l([(.3, .5), (.7, .5)]); l([(.3, .7), (.56 if kind == 'log' else .7, .7)])
    elif kind == 'checklist':
        bx(.16, .04, .68, .92); l([(.28, .34), (.36, .42), (.5, .26)]); l([(.56, .34), (.72, .34)]); l([(.28, .64), (.36, .72), (.5, .56)]); l([(.56, .64), (.72, .64)])
    elif kind == 'clipcheck': bx(.16, .1, .68, .86); bx(.34, .02, .32, .16, .8); l([(.3, .56), (.45, .7), (.72, .42)])
    elif kind == 'gauge': l(arc(.5, .7, .44, 180, 360)); l([(.5, .7), (.74, .4)]); l([(.06, .7), (.94, .7)])
    elif kind == 'flow':
        bx(.02, .06, .34, .26); bx(.64, .06, .34, .26); bx(.33, .68, .34, .26)
        l([(.36, .19), (.64, .19)]); l([(.19, .32), (.19, .5), (.5, .5), (.5, .68)]); l([(.81, .32), (.81, .5), (.5, .5)])
    elif kind == 'signal':
        l([(.5, .5), (.5, .96)]); o(.5, .42, .07)
        l(arc(.5, .42, .24, 135, 225, 8)); l(arc(.5, .42, .24, -45, 45, 8)); l(arc(.5, .42, .42, 140, 220, 8)); l(arc(.5, .42, .42, -40, 40, 8))
    elif kind == 'monitor':
        bx(.04, .08, .92, .6); l([(.5, .68), (.5, .84)]); l([(.28, .9), (.72, .9)]); l([(.14, .4), (.32, .4), (.4, .22), (.52, .56), (.6, .4), (.86, .4)])
    elif kind == 'target': o(.5, .5, .42); o(.5, .5, .2); l([(.5, 0), (.5, .2)]); l([(.5, .8), (.5, 1)]); l([(0, .5), (.2, .5)]); l([(.8, .5), (1, .5)])
    elif kind == 'bell':
        l([(.18, .76), (.26, .62), (.26, .42), (.34, .24), (.5, .16), (.66, .24), (.74, .42), (.74, .62), (.82, .76), (.18, .76)]); o(.5, .86, .07); l([(.5, .04), (.5, .16)])
    elif kind == 'stop':
        l([(.5 + .46 * math.cos(math.radians(22.5 + 45 * i)), .5 + .46 * math.sin(math.radians(22.5 + 45 * i))) for i in range(9)]); l([(.28, .5), (.72, .5)])
    elif kind == 'shield': l([(.5, .03), (.9, .18), (.85, .62), (.5, .97), (.15, .62), (.1, .18), (.5, .03)]); l([(.3, .5), (.45, .66), (.72, .36)])
    elif kind == 'cycle':
        # Clockwise arc open at the top; a filled head on the arc's tangent points back into the gap.
        a0, a1, rr = -50, 232, .38
        l(arc(.5, .52, rr, a0, a1, 24))
        t = math.radians(a1); ex, ey = .5 + rr * math.cos(t), .52 + rr * math.sin(t)
        tx, ty, nx, ny = -math.sin(t), math.cos(t), math.cos(t), math.sin(t); L, W = .3, .17
        d.poly(P([(ex + tx * L * .62, ey + ty * L * .62), (ex - tx * L * .38 + nx * W, ey - ty * L * .38 + ny * W), (ex - tx * L * .38 - nx * W, ey - ty * L * .38 - ny * W)]), fg)
    elif kind == 'chip':
        bx(.2, .2, .6, .6)
        for z in (.35, .5, .65): l([(z, .02), (z, .2)]); l([(z, .8), (z, .98)]); l([(.02, z), (.2, z)]); l([(.8, z), (.98, z)])
    elif kind == 'gateway': bx(.06, .52, .88, .34); l([(.3, .52), (.18, .1)]); l([(.7, .52), (.82, .1)]); o(.26, .69, .05); o(.46, .69, .05)
    elif kind == 'sensor':
        bx(.18, .14, .64, .72); l([(.28, .52), (.4, .52), (.47, .32), (.56, .7), (.63, .44), (.72, .44)])
        l([(.36, 0), (.36, .14)]); l([(.64, 0), (.64, .14)]); l([(.36, .86), (.36, 1)]); l([(.64, .86), (.64, 1)])
    elif kind == 'cloud': l(arc(.32, .58, .2, 90, 270, 10) + arc(.5, .42, .24, 200, 340, 10) + arc(.72, .58, .2, -90, 90, 10) + [(.32, .78)])
    elif kind == 'lock': bx(.18, .44, .64, .5); l(arc(.5, .44, .22, 180, 360, 12)); o(.5, .66, .06)
    elif kind == 'db':
        # Elliptical rims; sides and bottom as one path so the joins have no notches.
        rx, ry, top, bot = .38, .13, .16, .84
        l(ellipse(.5, top, rx, ry, -20, 380, 30)); l(ellipse(.5, .5, rx, ry, 0, 180, 12))
        l([(.5 - rx, top)] + ellipse(.5, bot, rx, ry, 180, 0, 16) + [(.5 + rx, top)])
    elif kind == 'gear':
        o(.5, .5, .26); o(.5, .5, .1)
        for i in range(8):
            ang = math.radians(i * 45); l([(.5 + .26 * math.cos(ang), .5 + .26 * math.sin(ang)), (.5 + .44 * math.cos(ang), .5 + .44 * math.sin(ang))])
    elif kind == 'server':
        for yy in (.06, .37, .68): bx(.1, yy, .8, .26, 1.5); o(.24, yy + .13, .035)
    elif kind == 'nn':
        # Sparse on purpose: denser layers clot below about 10 pt.
        cols = [[.3, .7], [.18, .5, .82], [.5]]
        xs_ = [.12, .5, .88]
        for (x1, c1), (x2, c2) in zip(zip(xs_, cols), zip(xs_[1:], cols[1:])):
            for a_ in c1:
                for b_ in c2: l([(x1, a_), (x2, b_)])
        for xx, cc in zip(xs_, cols):
            for yy in cc: o(xx, yy, .07)
    elif kind == 'chart':
        l([(.08, .06), (.08, .92), (.94, .92)]); bx(.2, .56, .14, .36, .5); bx(.44, .34, .14, .58, .5); bx(.68, .16, .14, .76, .5)
    elif kind == 'image':
        bx(.06, .14, .88, .72); l([(.12, .8), (.38, .48), (.56, .66), (.7, .52), (.88, .8)]); o(.7, .32, .07)
    elif kind == 'folder':
        l([(.06, .22), (.4, .22), (.48, .32), (.94, .32), (.94, .86), (.06, .86), (.06, .22)])
    elif kind == 'globe':
        o(.5, .5, .42); l([(.08, .5), (.92, .5)]); l(arc(.5, .5, .42, -90, 90, 12)); l([(.5, .08), (.5, .92)])
        l(arc(.5, .5, .2, -90, 90, 10)); l(arc(.5, .5, .2, 90, 270, 10))
    elif kind == 'bolt':
        l([(.58, .04), (.24, .56), (.5, .56), (.4, .96), (.78, .42), (.5, .42), (.58, .04)])
    else:
        raise ValueError('unknown glyph: ' + kind)

GLYPHS = ['patient', 'person', 'clinician', 'hospital', 'eye', 'network', 'stack', 'search', 'api', 'escalate', 'log', 'doc', 'checklist', 'clipcheck',
          'gauge', 'flow', 'signal', 'monitor', 'target', 'bell', 'stop', 'shield', 'cycle', 'chip', 'gateway', 'sensor', 'cloud', 'lock', 'db',
          'gear', 'server', 'nn', 'chart', 'image', 'folder', 'globe', 'bolt']

def tile(d, kind, x, y, s=32, bg=None):
    """A white line glyph on a dark rounded tile (Theme B/C icon style)."""
    bg = T.ramp[-1] if bg is None else bg
    d.box(x, y, s, s, bg, None, 3 * T.scale)
    glyph(d, kind, x + s * .2, y + s * .2, s * .6, WHITE, bg)

def checkmark(d, cx, cy, r=8, color=None):
    """Filled circle with a white tick: the shape partner of the 'control / check' colour."""
    color = T.ok if color is None else color
    d.circle(cx, cy, r, color, color, T.scale)
    d.line([(cx - r * .48, cy + r * .02), (cx - r * .1, cy + r * .4), (cx + r * .5, cy - r * .36)], WHITE, 1.7 * T.scale)

def warn(d, x, y, s=15, color=None):
    """Warning triangle, top-left at (x, y): the shape partner of the 'risk' colour."""
    color = T.risk if color is None else color
    d.poly([(x + s / 2, y), (x + s, y + s * .9), (x, y + s * .9)], color)
    d.line([(x + s / 2, y + s * .3), (x + s / 2, y + s * .58)], WHITE, 1.6 * T.scale)
    d.circle(x + s / 2, y + s * .72, .9 * T.scale, WHITE, WHITE, .6 * T.scale)

def badge(d, cx, cy, label, r=10, fill=None, size=None):
    """Numbered reading-order badge (Theme C), white bold label on a dark disc."""
    fill = T.extra.get('badge', T.ramp[-1]) if fill is None else fill; size = T.small if size is None else size
    d.circle(cx, cy, r, fill, fill, T.scale)
    d.text(cx - r, cy - size * .62, 2 * r, str(label), size, True, WHITE, 'center')

def letter_tile(d, x, y, s, letter, size=None, fill=None):
    """Dark square with a white letter (dimension codes such as S/A/F/E/T, layer codes L1-L5)."""
    fill = T.ramp[-1] if fill is None else fill; size = 21 * T.scale if size is None else size
    d.box(x, y, s, s, fill, None, 3 * T.scale)
    d.ctext(x, y, s, s, letter, size, True, WHITE)

def panel_letter(d, x, y, i, heading=None):
    """Panel label in the theme's convention: 'a' bold lowercase (A), '(a)' (C), or nothing (B)."""
    if T.panel_letters == 'none': return
    lab = 'abcdefghij'[i] if T.panel_letters == 'a' else f'({"abcdefghij"[i]})'
    d.text(x, y, 40, lab, T.zone, True)
    if heading: d.text(x + (22 if T.panel_letters == 'a' else 36), y + .8, 960 - x, heading, T.title, True)

# ---- output ------------------------------------------------------------------
def check_errors():
    if ERRORS: raise SystemExit('Layout errors:\n' + '\n'.join(ERRORS))

def render(pdf, dpi=220, gray=True):
    """PNG (and greyscale PNG) beside the PDF. Uses ghostscript, else PyMuPDF."""
    pdf = Path(pdf); outs = []
    for dev, suffix in [('png16m', ''), ('pnggray', '_gray')] if gray else [('png16m', '')]:
        png = pdf.with_name(pdf.stem + suffix + '.png'); outs.append(png)
        if shutil.which('gs'):
            subprocess.run(['gs', '-q', '-dNOPAUSE', '-dBATCH', f'-sDEVICE={dev}', f'-r{dpi}', '-dTextAlphaBits=4', '-dGraphicsAlphaBits=4',
                            f'-sOutputFile={png}', str(pdf)], check=True)
        else:
            try: import pymupdf as fitz
            except ImportError: import fitz
            page = fitz.open(pdf)[0]
            page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), colorspace=fitz.csGRAY if suffix else fitz.csRGB).save(str(png))
    return outs

def contact_sheet(pngs, out, pad=30, bg='#D5DAE0', max_width=None):
    """Stack PNGs vertically on one sheet (for showing a whole figure set or comparing styles)."""
    from PIL import Image
    ims = [Image.open(p).convert('RGB') for p in pngs]
    if max_width:
        ims = [im.resize((max_width, int(im.height * max_width / im.width))) if im.width > max_width else im for im in ims]
    W = max(i.width for i in ims)
    sheet = Image.new('RGB', (W + 2 * pad, sum(i.height for i in ims) + pad * (len(ims) + 1)), bg)
    y = pad
    for im in ims: sheet.paste(im, (pad, y)); y += im.height + pad
    sheet.save(out)
    return out

def stats_table():
    rows = [f"{f['stem']:<34} words={f['words']:>4}  min={f['min_pt']:.2f} pt  height={f['height_pt']} pt" for f in ALL]
    return '\n'.join(rows)

use_theme('B')
