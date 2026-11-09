# Vendored from the authors' paper-figures toolkit (figure helpers used by paper/figures/make_figures.py).
# Released here under the MIT License, as the rest of this repository (see LICENSE).
"""Style presets shared by figkit (diagrams, ReportLab) and plotkit (data plots, matplotlib).

No third-party dependencies, so plot scripts can import it without ReportLab.
A: clinical editorial; B: slate + one accent; C: mechanism-first threat map.
See references/styles.md for when to use which.
"""
from dataclasses import dataclass, field

@dataclass
class Theme:
    name: str
    font: tuple
    ink: str; muted: str; line: str; panel: str
    ramp: list                       # structure / ordinal steps, light to dark
    ok: str; ok_tint: str            # controls, checks, governance
    risk: str; risk_tint: str        # failure, fault, risk
    accent: str; accent_tint: str    # one highlight (data, "ours"), or the user/system role in C
    extra: dict = field(default_factory=dict)
    r_outer: float = 8; r_inner: float = 3
    lw_box: float = 1.0; lw_conn: float = 1.45; lw_loop: float = 2.4
    body: float = 16.5; item: float = 16; small: float = 15.5; title: float = 17.5; zone: float = 18.5
    panel_letters: str = 'none'      # 'none', 'a' (bold lowercase), '(a)'
    scale: float = 1.0               # set by figkit.use_theme: units per unit at 516 pt, so strokes print at the same thickness at any width

THEMES = {
    # A: Nature Medicine / Lancet / BMJ. Square corners, keylines, near-black text never coloured, pale fills with one meaning each, no icons.
    'A': Theme('clinical-editorial', ('arial', 'helvetica-neue', 'liberation', 'dejavu'),
               ink='#231F20', muted='#4A4A4A', line='#3C4E6C', panel='#DEE4EA',
               ramp=['#EEF1F4', '#DEE4EA', '#C9D2DD', '#B3BECC', '#9CA8BA', '#3C4E6C'],
               ok='#3C6E47', ok_tint='#E3F1DC', risk='#9A3B2E', risk_tint='#F8C4BA', accent='#3C4E6C', accent_tint='#8AC6EA',
               extra={'neutral': '#DEDEC6', 'evidence': '#FFF6D2'},
               r_outer=0, r_inner=0, lw_box=1.1, lw_conn=1.1, lw_loop=1.8, body=16.5, item=16.5, small=15.5, title=17.5, zone=18.5,
               panel_letters='a'),
    # B: Nature 2026 LLM-safety Fig. 1 / GMAI / AMIE / IEEE COMST. One slate ramp, order by lightness, teal only for controls, warm only for risk.
    'B': Theme('slate-one-accent', ('helvetica-neue', 'arial', 'liberation', 'dejavu'),
               ink='#1F2A36', muted='#4F5D6E', line='#B2BACB', panel='#EEF1F5',
               ramp=['#EEF1F5', '#D9DFE7', '#B2BACB', '#8B97AD', '#61708A', '#3D4F6A'],
               ok='#237668', ok_tint='#E3EFEC', risk='#A74835', risk_tint='#F6E5DF', accent='#237668', accent_tint='#E3EFEC',
               extra={'layer': ['#E3E8EE', '#D9DFE7', '#C6CEDA', '#B2BACB', '#8B97AD']}),
    # C: AgentDojo / Agent Security Bench / CaMeL / USENIX. Neutral structure, colour by role with a shape backup, numbered badges, dashed trust boundaries.
    'C': Theme('mechanism-first', ('helvetica-neue', 'arial', 'liberation', 'dejavu'),
               ink='#2B2B2B', muted='#5F6670', line='#2B2B2B', panel='#F3F4F6',
               ramp=['#F3F4F6', '#E3E5E8', '#C9CDD2', '#9AA0A8', '#59616C', '#3A3F47'],
               ok='#2E7D4F', ok_tint='#E6F2EA', risk='#B3382C', risk_tint='#FBEBE9', accent='#1F5F99', accent_tint='#E7EEF6',
               extra={'badge': '#3A3F47', 'layer': ['#F3F4F6', '#E9EBEE', '#DEE1E5', '#D2D6DB', '#C5CAD0']},
               r_outer=3, r_inner=3, lw_box=1.45, lw_conn=1.45, lw_loop=2.4, body=17.5, item=17, small=15.5, title=19.5, zone=19.5,
               panel_letters='(a)'),
}
WHITE = '#FFFFFF'

def luminance(hexcolor):
    rgb = [int(hexcolor.lstrip('#')[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    lin = [c / 12.92 if c <= .04045 else ((c + .055) / 1.055) ** 2.4 for c in rgb]
    return .2126 * lin[0] + .7152 * lin[1] + .0722 * lin[2]

def on(fill, ink='#1F2A36'):
    """Text colour for a fill: white on dark fills, ink otherwise (keeps contrast at 4.5:1 or better)."""
    return WHITE if 1.05 / (luminance(fill) + .05) >= 4.5 else ink
