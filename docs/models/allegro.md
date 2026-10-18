# Allegro

The Wonik Robotics Allegro V3 has 21 skeleton joints and 16 independent
`hand_pose` coordinates.

Source: MuJoCo Menagerie `wonik_allegro` (BSD-2-Clause).
Assets download automatically; see [asset setup](../index.md#assets).

A manual `model_path` is a directory with `left.xml`, `right.xml` and their meshes.

```python
from robot_models.hands.allegro.numpy import AllegroHand

hand = AllegroHand(side="right")
params = hand.get_rest_pose(preset="rest")
```

The hand has four fingers and no coupled joints. The common wrist joint is the palm origin at the finger bases; the upstream model has no wrist body.

## API

::: robot_models.hands.allegro.numpy.AllegroHand
