# Render teasers

Render separate hand and humanoid lineups from the actual meshes, with
transparent PNG backgrounds:

```bash
uv run --script scripts/render_teaser.py --output-dir docs/assets/teasers
```

`uv` creates an isolated Python 3.13/Blender environment that uses the current
checkout. Assets download on first use. By default, images go to `results/teasers/`.

## Options

```bash
# Select a lineup and models.
uv run --script scripts/render_teaser.py --group hands --models allegro shadow

# Render on an OptiX-capable NVIDIA GPU and save editable Blender scenes.
uv run --script scripts/render_teaser.py --device gpu --samples 256 --save-blend
```

Use `--group humanoids` for the humanoid lineup. Set `--width`, `--height`,
`--samples` and `--threads` as needed. CPU rendering with eight threads is the
default. `--save-blend` writes scenes beside the PNGs.

Humanoids use T-pose presets; hands use right-side flat presets. Models have
equal display heights, so the images do not compare physical size. The Shadow
forearm is hidden to show the palm and fingers. Asset licenses also apply to
redistributed renders and scenes; see [hand sources](hand-sources.md).
