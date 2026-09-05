"""
Generate the ReservoirX-D logo.

The mark is not decorative: it is an actual directed 3-regular digraph produced
by app/core/graphs.py. Every node has exactly three outgoing and three incoming
edges, which is the property the whole service is built on. Edges taper from
thick at the source to thin at the target, so direction reads without arrowheads
at small sizes.

Renders at 4x and downsamples for antialiasing.
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.core.graphs import (  # noqa: E402
    edges_to_matrix,
    regular_degree_sequence,
    simple_digraph_from_degrees,
)

OUT = pathlib.Path(__file__).resolve().parents[1] / "assets"
FINAL = 500
SCALE = 4
SIZE = FINAL * SCALE

BACKGROUND = (11, 17, 32)
EDGE_START = (34, 211, 238)   # cyan
EDGE_END = (129, 140, 248)    # violet
NODE_FILL = (241, 245, 249)
NODE_GLOW = (56, 189, 248)


def bezier(p0, p1, p2, steps: int):
    t = np.linspace(0.0, 1.0, steps)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2


def lerp(a, b, t: float):
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def draw_tapered_edge(draw, p_from, p_to, centre, width_from, width_to, curvature=0.42):
    mid = (p_from + p_to) / 2.0
    ctrl = centre + (mid - centre) * (1.0 - curvature)
    pts = bezier(p_from, ctrl, p_to, 90)

    for i in range(len(pts) - 1):
        t = i / (len(pts) - 2)
        width = width_from + (width_to - width_from) * t
        colour = lerp(EDGE_START, EDGE_END, t)
        alpha = int(200 - 70 * t)
        draw.line(
            [tuple(pts[i]), tuple(pts[i + 1])],
            fill=colour + (alpha,),
            width=max(1, int(round(width))),
        )


def build(n_nodes: int = 10, degree: int = 3, seed: int = 5) -> Image.Image:
    rng = np.random.default_rng(seed)
    seq = regular_degree_sequence(n_nodes, degree)
    edges = simple_digraph_from_degrees(seq, seq, rng)

    A = edges_to_matrix(edges, n_nodes)
    assert np.all(A.sum(axis=1) == degree) and np.all(A.sum(axis=0) == degree), (
        "the logo must depict a genuinely degree-regular digraph"
    )

    centre = np.array([SIZE / 2.0, SIZE / 2.0])
    radius = SIZE * 0.335
    angles = [(-math.pi / 2) + 2 * math.pi * i / n_nodes for i in range(n_nodes)]
    positions = np.array(
        [[centre[0] + radius * math.cos(a), centre[1] + radius * math.sin(a)] for a in angles]
    )

    # --- background with a soft radial lift ------------------------------
    image = Image.new("RGB", (SIZE, SIZE), BACKGROUND)
    glow_bg = Image.new("RGB", (SIZE, SIZE), (0, 0, 0))
    d = ImageDraw.Draw(glow_bg)
    r = SIZE * 0.44
    d.ellipse([centre[0] - r, centre[1] - r, centre[0] + r, centre[1] + r], fill=(18, 32, 60))
    image = ImageChops.add(image, glow_bg.filter(ImageFilter.GaussianBlur(SIZE * 0.10)))

    # --- edges ------------------------------------------------------------
    edge_layer = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    edge_draw = ImageDraw.Draw(edge_layer)
    for u, v in edges:
        draw_tapered_edge(
            edge_draw, positions[u], positions[v], centre,
            width_from=SCALE * 5.4, width_to=SCALE * 1.1,
        )

    # additive bloom around the edges, then the crisp edges on top
    bloom = edge_layer.convert("RGB").filter(ImageFilter.GaussianBlur(SCALE * 5))
    image = ImageChops.add(image, ImageChops.multiply(bloom, Image.new("RGB", (SIZE, SIZE), (120, 120, 120))))
    image = Image.alpha_composite(image.convert("RGBA"), edge_layer).convert("RGB")

    # --- nodes: blurred halo, then solid core ----------------------------
    node_r = SCALE * 12.0
    halo = Image.new("RGB", (SIZE, SIZE), (0, 0, 0))
    hd = ImageDraw.Draw(halo)
    for x, y in positions:
        hr = node_r * 2.4
        hd.ellipse([x - hr, y - hr, x + hr, y + hr], fill=NODE_GLOW)
    image = ImageChops.add(image, halo.filter(ImageFilter.GaussianBlur(SCALE * 9)))

    core = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    cd = ImageDraw.Draw(core)
    for x, y in positions:
        cd.ellipse(
            [x - node_r * 1.5, y - node_r * 1.5, x + node_r * 1.5, y + node_r * 1.5],
            fill=BACKGROUND + (235,),
        )
        cd.ellipse([x - node_r, y - node_r, x + node_r, y + node_r], fill=NODE_FILL + (255,))
    image = Image.alpha_composite(image.convert("RGBA"), core).convert("RGB")

    return image.resize((FINAL, FINAL), Image.LANCZOS)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    logo = build()

    png = OUT / "logo-500.png"
    logo.save(png, "PNG", optimize=True)

    jpg = OUT / "logo-500.jpg"
    logo.save(jpg, "JPEG", quality=92, optimize=True)

    logo.resize((256, 256), Image.LANCZOS).save(OUT / "logo-256.png", "PNG", optimize=True)
    logo.resize((64, 64), Image.LANCZOS).save(OUT / "logo-64.png", "PNG", optimize=True)

    for path in sorted(OUT.glob("logo-*")):
        with Image.open(path) as im:
            print(f"{path.name:16s} {im.size[0]}x{im.size[1]}  {path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
