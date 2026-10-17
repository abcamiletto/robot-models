# Render model teasers

`scripts/render_teaser.py` follows the
[Body Models teaser](https://github.com/abcamiletto/body-models/blob/main/scripts/render_teaser.py):
load actual meshes through the NumPy API and render labeled pastel lineups with
headless Blender/Cycles. It produces separate humanoid and hand PNGs.

![Humanoid lineup](assets/teasers/humanoids.png)

![Hand lineup](assets/teasers/hands.png)

From a robot-models checkout:

```bash
uv run --script scripts/render_teaser.py
```

The script declares its own Python 3.13 and Blender dependencies using PEP 723.
`uv` provisions that isolated environment; Blender is not added to the library's
runtime or development dependencies. The local library is installed editable so
renders use the current checkout. Published assets download on first use,
following normal `robot-models` configuration and cache conventions.

The default output is `results/teasers/humanoids.png` and `results/teasers/hands.png`:

- Humanoids: G1, GR1, H1, T1 and TALOS in their T-pose presets.
- Hands: right Ability, Allegro, BrainCo, Inspire, LEAP, Sharpa and Shadow in their
  flat presets. Each palm faces the camera with fingers pointing upward.

Meshes are rotated and scaled for display, with equal heights within each lineup.
The images are **not a comparison of physical size**. Shadow's long forearm mesh
is omitted in hand teasers so the palm and fingers remain readable. Model assets,
poses and runtime defaults are unchanged. A proper rotation converts Y-up model
coordinates to Blender Z-up while preserving triangle winding.

## Rendering options

```bash
# Fast draft, using an NVIDIA GPU with Cycles OptiX.
uv run --script scripts/render_teaser.py --device gpu --samples 16 --width 1600 --height 500

# High-quality transparent images and editable Blender scene files.
uv run --script scripts/render_teaser.py --device gpu --samples 256 --save-blend

# Render one lineup, or choose a subset.
uv run --script scripts/render_teaser.py --group humanoids
uv run --script scripts/render_teaser.py --group hands --models allegro shadow

# Regenerate the images committed in the README and documentation.
uv run --script scripts/render_teaser.py --output-dir docs/assets/teasers
```

CPU rendering is the default and uses eight threads; change this with `--threads`.
`--device gpu` requires an OptiX-capable NVIDIA GPU and fails clearly if none is
available. All exports are RGBA PNGs with transparent backgrounds. Neutral gray
labels work on both light and dark pages. The output directory, width, height and
sample count are configurable. Camera framing includes all meshes and labels.

`--save-blend` writes `humanoids.blend` and/or `hands.blend` alongside the PNGs,
including the model meshes, materials, lights, camera and render configuration.
Generated PNGs and scenes stay in the ignored `results/` directory by default.

## Original baseline hands

`--models` accepts all registered models of the selected group, including the
[original grasp baseline variants](models/baselines.md). Prepare and configure
their local assets first; the renderer does not substitute or silently omit models:

```bash
robot-models set barrett-gendexgrasp /path/to/barrett-gendexgrasp/assets
robot-models set robotiq-3finger-gendexgrasp /path/to/robotiq-3finger-gendexgrasp/assets
robot-models set ezgripper-gendexgrasp /path/to/ezgripper-gendexgrasp/assets
uv run --script scripts/render_teaser.py --group hands \
  --models barrett-gendexgrasp robotiq-3finger-gendexgrasp ezgripper-gendexgrasp
```

Select a Shadow source through its single model family:

```bash
uv run --script scripts/render_teaser.py --group hands --models shadow --shadow-source dexgraspnet
```

`--shadow-source` accepts `menagerie` (default), `gendexgrasp`, or `dexgraspnet`.
Sources retain their native side; EZGripper uses its original left assembly.
These render the variants' display geometry, not their collision or contact
algorithms. Asset terms still apply to any images or scenes you redistribute.
