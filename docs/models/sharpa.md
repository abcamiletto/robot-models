# Sharpa Wave

Sharpa is the Sharpa Wave dexterous hand. Each standard hand has 23 skeleton
joints (one per MJCF body) and 22 independent `hand_pose` coordinates: five
thumb hinges, four hinges per finger, and an extra pinky CMC hinge.

## Setup

Assets are derived from the official
[`sharpa-robotics/sharpa-urdf-usd-xml`](https://github.com/sharpa-robotics/sharpa-urdf-usd-xml)
repository at revision `0d19cac602f46456b819e4b6a2c09a74982c9a3e`, licensed
under Apache 2.0. The prepared archive preserves the upstream license and notices.
The standard palm-rooted models are used; wrist and flange variants are excluded.

Assets download from `sharpa/assets.zip` in
[`abcamiletto/robot-models`](https://huggingface.co/abcamiletto/robot-models).
Assets download on first use, or with:

```bash
robot-models download sharpa
```

To reproduce the hosted assets locally:

```bash
git clone https://github.com/sharpa-robotics/sharpa-urdf-usd-xml /tmp/sharpa-upstream
git -C /tmp/sharpa-upstream checkout 0d19cac602f46456b819e4b6a2c09a74982c9a3e
uv run python scripts/prepare_sharpa_assets.py /tmp/sharpa-upstream /tmp/sharpa-hub
mkdir -p /tmp/sharpa-model
unzip /tmp/sharpa-hub/assets.zip -d /tmp/sharpa-model
robot-models set sharpa /tmp/sharpa-model
```

A manual `model_path` points to the extracted directory containing `left.xml`,
`right.xml`, and the two mesh folders. The archive and `LICENSE.md` preserve upstream licensing and record the path changes.

## Usage

```python
from robot_models.hands.sharpa.numpy import SharpaHand

hand = SharpaHand(side="right")  # also supports side="left"
params = hand.get_rest_pose(hands="rest")
skeleton = hand.forward_skeleton(**params)
meshes = hand.forward_meshes(**params)
```

Equivalent classes are available in `robot_models.hands.sharpa.torch` and
`robot_models.hands.sharpa.jax`.

## Notes

The `flat` preset is the zero joint pose. The `rest` preset uses 0.2 radians
for flexion hinges (CMC FE, MCP FE, IP, PIP, and DIP), 0.05 radians for the
pinky CMC, and zero for abduction/adduction. These are package reference poses,
not manufacturer-calibrated resting poses.

Common anatomical joints map to the physical finger segments: thumb MC, PP,
and DP, and finger PP, MP, and DP. The additional virtual links for the
two-axis CMC and MCP joints remain available in the full skeleton.

## API

::: robot_models.hands.sharpa.numpy.SharpaHand
