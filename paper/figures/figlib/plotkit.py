# Vendored from the authors' paper-figures toolkit (figure helpers used by paper/figures/make_figures.py).
# Released here under the MIT License, as the rest of this repository (see LICENSE).
"""plotkit: matplotlib settings that match a figkit theme at the printed width.

    import sys; sys.path.insert(0, '<skill>/scripts')
    import matplotlib.pyplot as plt, plotkit
    plt.rcParams.update(plotkit.mpl_style('B'))
    fig, ax = plt.subplots(figsize=plotkit.figsize(252, 0.62))   # 252 pt = one IEEE column

Needs matplotlib only (no ReportLab).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from themes import THEMES

def _family_with_bold(prefs):
    """First family in prefs that matplotlib can render in both regular and bold.

    On macOS matplotlib registers only the regular face of HelveticaNeue.ttc, so bold text would silently
    print regular; Arial (separate Regular and Bold files) is the safe Helvetica-class choice for plots."""
    from matplotlib import font_manager as fm
    def is_bold(f):
        w = f.weight
        return (isinstance(w, int) and w >= 600) or (isinstance(w, str) and w.lower() in ('bold', 'semibold', 'demibold', 'heavy', 'black'))
    have = {f.name for f in fm.fontManager.ttflist}
    bold = {f.name for f in fm.fontManager.ttflist if is_bold(f)}
    for name in prefs:
        if name in have and name in bold: return name
    return 'DejaVu Sans'

def mpl_style(theme=None):
    """rcParams for matplotlib plots matching a theme. Sizes are absolute (8-9 pt), so set the figure size
    to the printed width with figsize() and include the PDF without scaling.

    usage: plt.rcParams.update(plotkit.mpl_style('B')); fig, ax = plt.subplots(figsize=plotkit.figsize(252, 0.62))"""
    th_ = THEMES[theme or 'B'] if isinstance(theme, (str, type(None))) else theme
    family = _family_with_bold(['Arial', 'Helvetica', 'Helvetica Neue', 'Liberation Sans', 'DejaVu Sans'])
    from cycler import cycler
    series = [th_.ramp[-1], th_.ok, th_.risk, th_.ramp[3], th_.accent if th_.accent != th_.ok else th_.ramp[4], th_.ramp[2]]
    return {
        'font.family': 'sans-serif', 'font.sans-serif': [family, 'DejaVu Sans'],
        'font.size': 8.5, 'axes.titlesize': 9, 'axes.labelsize': 8.5, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
        'axes.edgecolor': th_.ink, 'axes.labelcolor': th_.ink, 'xtick.color': th_.ink, 'ytick.color': th_.ink, 'text.color': th_.ink,
        'axes.linewidth': .6, 'xtick.major.width': .6, 'ytick.major.width': .6, 'xtick.major.size': 3, 'ytick.major.size': 3,
        'lines.linewidth': 1.4, 'lines.markersize': 4, 'patch.linewidth': .5,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': False, 'grid.color': th_.ramp[1], 'grid.linewidth': .5,
        'axes.prop_cycle': cycler(color=series) + cycler(marker=['o', 's', '^', 'D', 'v', 'P']),
        'legend.frameon': False, 'figure.dpi': 150, 'savefig.dpi': 600,
        # Keep the saved page exactly figsize: a 'tight' bbox grows the page around outside labels, and LaTeX then shrinks the type.
        'savefig.bbox': None, 'figure.constrained_layout.use': True,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    }

def figsize(width_pt, aspect=0.6):
    """(width, height) in inches for a plot printed at width_pt; aspect = height / width."""
    return (width_pt / 72, width_pt / 72 * aspect)

