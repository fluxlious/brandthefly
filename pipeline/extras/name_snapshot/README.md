# Name snapshot (first experiment)

Writes letters into the brain (still, not sliding), simulates it, and renders a PNG snapshot plus a GIF.

```bash
cd pipeline/extras/name_snapshot
python sim.py 300 0 4 400      # font size, y offset, cell size (µm), duration (ms)
python render.py               # -> snapshot.png, animation.gif
```
The text is set in `sim.py` (the `'CEM'` letters). Needs steps 00–03 of the pipeline first.
