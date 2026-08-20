"""
split_figure2.py -- make Figure 2 (the DeepS3++ / soil-climate-landscape module)
legible in print.

The submitted figure is 1186 px wide and is placed at \textwidth, so its block
labels render at about 4 pt, which is what Reviewer 1 (Minor 1) objected to.
No vector source exists, so instead of redrawing (and risking a change to the
architecture it depicts) the original artwork is cut at its three natural
whitespace corridors and the pieces are stacked. Each piece is then shown at a
width that puts its labels at 8 pt or larger.

Cut points come from a column-ink scan of the original: x = 215..222 and
x = 789..800 are the only interior columns with essentially no ink.
"""
import os
import numpy as np
from PIL import Image

SRC = "/Users/a/Claude/Projects/Paper 3 PhD/manuscript/paper3/sclmodule.png"
OUT = os.path.dirname(os.path.abspath(__file__))
UPSCALE = 3.5          # keeps the panels near 300 dpi at their printed width
PAD = 6                # px of breathing room either side of a cut

# cuts sit inside the empty corridors, so no padding is applied across a cut
PANELS = [("a", 0, 219), ("b", 221, 795), ("c", 797, 1186)]


def main():
    im = Image.open(SRC).convert("RGBA")
    a = np.array(im)
    ink = (a[..., 3] > 10) & (a[..., :3].min(axis=2) < 230)

    for name, x0, x1 in PANELS:
        sub = ink[:, x0:x1]
        rows = np.nonzero(sub.sum(axis=1))[0]
        y0, y1 = max(0, rows.min() - PAD), min(im.height, rows.max() + 1 + PAD)

        panel = im.crop((x0, y0, x1, y1))
        if name == "a":
            # rows 274-316 of the original are just the vertical connector arrow;
            # shorten it so the panel is less elongated. The arrow is a straight
            # line, so removing a segment is visually seamless.
            keep = 10
            b0, b1 = 274 - y0, 316 - y0
            top = panel.crop((0, 0, panel.width, b0 + keep))
            bot = panel.crop((0, b1, panel.width, panel.height))
            merged = Image.new("RGBA", (panel.width, top.height + bot.height))
            merged.paste(top, (0, 0))
            merged.paste(bot, (0, top.height))
            panel = merged
        # flatten onto white so the PDF has no transparency surprises
        flat = Image.new("RGB", panel.size, "white")
        flat.paste(panel, mask=panel.split()[3])
        big = flat.resize((int(flat.width * UPSCALE), int(flat.height * UPSCALE)),
                          Image.LANCZOS)
        path = os.path.join(OUT, "figure2_%s.png" % name)
        big.save(path, dpi=(300, 300))
        print("panel %s: crop x %d-%d y %d-%d -> %dx%d px  (%s)"
              % (name, x0, x1, y0, y1, big.width, big.height, os.path.basename(path)))


if __name__ == "__main__":
    main()
