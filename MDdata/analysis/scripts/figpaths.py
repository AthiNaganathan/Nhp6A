"""Where a figure lives under figures/ -- the single routing rule for the project.

figures/ is split into one subfolder per system, named after the system prefix of
the filename (the <system> of <system>_<observable>[_<variant>]_v<N>.<ext>):

  figures/1Nhp6A_apo/   apo WT, small and large box   (1Nhp6A_apo*)
  figures/1Nhp6A/       1:1 WT complex, stiff + wall + equil + m-state (1Nhp6A_*)
  figures/2Nhp6A/       2:1 WT complex                (2Nhp6A_*)
  figures/1S26D/, 1T63D/, 1TM/   phosphomimetic mutants (incl. their _apo/_large)
  figures/1IDR/, 1D2_10/, DNA/   IDR-only, deletion, free DNA
  figures/              figures that combine systems (mutants_*, pmf_*_compare_*,
                        all_*, *_vs_dna*, apo_vs_holo, ...) and unattributable legacy
  figures/article/      article panels (notebooks/article/, unchanged)
  figures/preliminary/  retired early-phase figures (pre-_v<N>, pre-wall), flat;
                        moved there by --archive, never written to by figpath()

Every script/notebook builds its output path with figpath(name), so the rule is
written once.  Usage:

    from figpaths import figpath
    fig.savefig(figpath("1Nhp6A_wall_pmf_bend_v2.png"), dpi=300)
    figpath(name, root=ROOT)          # absolute, for scripts that use ROOT

`python analysis/scripts/figpaths.py --migrate [--apply]` moves files that sit in
the wrong folder (dry run unless --apply) and appends old->new to
figures/MOVED.tsv, so names quoted in LABNOTEBOOK.md stay resolvable.
`--archive LIST [--apply]` retires the listed files into figures/preliminary/.
"""
import os
import re
import sys

FIGROOT = "figures"


# Multi-system figures that carry one system's prefix: they stay in figures/.
# Checked before the prefix table.
MULTI_PATTERNS = [
    r"_vs_dna",            # 1Nhp6A_wall_pmf_*_vs_dna*  (WT vs free DNA)
    r"_apo_vs_holo",       # 1Nhp6A_equil_rmsf_apo_vs_holo
    r"_apo_holo",          # 1Nhp6A_mindist_pi_apo_holo
    r"_bend_compare",      # 2Nhp6A_equil_bend_compare  (2:1 vs 1:1)
    r"_mmcompare",         # 2Nhp6A_mstate_md_mmcompare (1:1 m-state vs 2:1 copy B)
]

# (filename prefix, subfolder); first match wins, so longer prefixes go first.
PREFIX_DIRS = [
    ("1Nhp6A_apo", "1Nhp6A_apo"),
    ("1Nhp6A_", "1Nhp6A"),
    ("1Nh6pA_", "1Nhp6A"),       # legacy typo
    ("2Nhp6A_", "2Nhp6A"),
    ("1S26D_", "1S26D"),
    ("1S2D_", "1S26D"),          # legacy typo
    ("S26D_", "1S26D"),
    ("1T63D_", "1T63D"),
    ("1TM_", "1TM"),
    ("1IDR_", "1IDR"),
    ("1D2_10_", "1D2_10"),
    ("DNA_", "DNA"),
    ("replicas_", "DNA"),        # notebooks/analysis_replicas.ipynb: DNA 6-walker MetaD
]

SUBDIRS = sorted({d for _, d in PREFIX_DIRS})
PRELIM = "preliminary"   # retired early-phase figures (flat); never routed to, see --archive


def figdir(name):
    """Subfolder of figures/ for a figure filename ('' = figures/ itself)."""
    base = os.path.basename(name)
    if any(re.search(p, base) for p in MULTI_PATTERNS):
        return ""
    for prefix, sub in PREFIX_DIRS:
        if base.startswith(prefix):
            return sub
    return ""


def figpath(name, root=None, makedirs=True):
    """figures/<subfolder>/<name> (prefixed by root if given); creates the folder."""
    d = os.path.join(root or "", FIGROOT, figdir(name))
    if makedirs:
        os.makedirs(d, exist_ok=True)
    return os.path.join(d, os.path.basename(name))


def figout(name, figdir=None, root=None):
    """figpath(name), unless a script's explicit --figdir overrides the routing."""
    if figdir is None:
        return figpath(name, root=root)
    os.makedirs(figdir, exist_ok=True)
    return os.path.join(figdir, name)


def _move(root, moves, apply):
    """Move (old, new) root-relative pairs, refusing clashes; log to figures/MOVED.tsv."""
    log = os.path.join(root, FIGROOT, "MOVED.tsv")
    clash = [n for _, n in moves if os.path.exists(os.path.join(root, n))]
    if clash:
        sys.exit("refusing: targets already exist:\n  " + "\n  ".join(clash))
    for o, n in moves:
        print(f"{o} -> {n}")
    print(f"{len(moves)} file(s) {'moved' if apply else 'to move (dry run)'}")
    if apply and moves:
        import datetime
        new = not os.path.exists(log)
        today = datetime.date.today().isoformat()
        with open(log, "a") as fh:
            if new:
                fh.write("# old_path\tnew_path\tdate  (written by figpaths.py)\n")
            for o, n in moves:
                os.makedirs(os.path.dirname(os.path.join(root, n)), exist_ok=True)
                os.rename(os.path.join(root, o), os.path.join(root, n))
                fh.write(f"{o}\t{n}\t{today}\n")
        print(f"logged in {os.path.relpath(log, root)}")


def _migrate(root, apply):
    """Files in figures/ or a system folder that figdir() routes elsewhere.
    figures/preliminary/ and figures/article/ itself are never scanned."""
    top = os.path.join(root, FIGROOT)
    moves = []
    for sub in [""] + SUBDIRS:
        d = os.path.join(top, sub)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not os.path.isfile(os.path.join(d, f)) or f == "MOVED.tsv":
                continue
            want = figdir(f)
            if want != sub:
                moves.append((os.path.join(FIGROOT, sub, f),
                              os.path.join(FIGROOT, want, f)))
    _move(root, moves, apply)


def _archive(root, paths, apply):
    """Retire figures (paths relative to figures/) into figures/preliminary/, flat."""
    moves = []
    for p in paths:
        p = p[len(FIGROOT) + 1:] if p.startswith(FIGROOT + "/") else p
        if not os.path.isfile(os.path.join(root, FIGROOT, p)):
            sys.exit(f"no such file: {FIGROOT}/{p}")
        moves.append((os.path.join(FIGROOT, p),
                      os.path.join(FIGROOT, PRELIM, os.path.basename(p))))
    _move(root, moves, apply)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--migrate", action="store_true",
                    help="list (or with --apply, move) misplaced files in figures/")
    ap.add_argument("--archive", metavar="LIST",
                    help="file listing figures/ paths (one per line) to retire into "
                         "figures/preliminary/ (dry run unless --apply)")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    ap.add_argument("names", nargs="*", help="print the routed path of these names")
    a = ap.parse_args()
    for n in a.names:
        print(figpath(n, makedirs=False))
    if a.migrate:
        _migrate(a.root, a.apply)
    if a.archive:
        _archive(a.root, [l.strip() for l in open(a.archive) if l.strip()
                          and not l.startswith("#")], a.apply)
