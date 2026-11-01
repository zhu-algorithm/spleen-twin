"""Regenerate docs/index.html from a fresh simulation run.

The site is a single self-contained HTML file: no CDN, no build step, no
network at runtime. Every number in it comes from one seeded run of the
twin, embedded as JSON at build time, so the page and the simulation can
never drift apart.

Usage:  python scripts/build_site.py --seed 20261030
"""
