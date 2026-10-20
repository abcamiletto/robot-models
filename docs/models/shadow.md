# Shadow

The Shadow Dexterous Hand E3M5 starts at the forearm. It has 25 skeleton
joints and 24 independent `hand_pose` coordinates.

Source: MuJoCo Menagerie `shadow_hand` (Apache-2.0).
Assets download automatically; see [asset setup](../index.md#assets).

A manual `model_path` is a directory with `left.xml`, `right.xml`, `right-openai.xml` and their meshes.

```python
from robot_models.hands.shadow.numpy import ShadowHand

hand = ShadowHand(side="right")
params = hand.get_rest_pose(preset="rest")
```

Every hinge is independent: the J1/J2 tendon coupling of the real hand is actuation, not kinematics.

## OpenAI kinematics

`kinematics="openai"` loads the right hand with the kinematics of OpenAI Gym's Shadow hand
(`robot0:`, from its hand-manipulation environments; Apache-2.0), the hand DexGraspNet
optimizes grasps with. The bodies, joints, coordinates, meshes and API are the default's:

```python
hand = ShadowHand(side="right", kinematics="openai")
```

The two descriptions are the same hand except for two joint placements and the joint ranges.
The default follows Shadow Robot's own descriptions:

| | Default (Menagerie, Shadow Robot) | `"openai"` |
| --- | --- | --- |
| `rh_LFJ5`, little-finger metacarpal | at (-33, 0, 20.71) mm in the palm, axis (0.573576, 0, 0.819152) | at (-17, 0, 44) mm, axis (0.571, 0, 0.821) normalized |
| `rh_lfknuckle` at rest | (-33, 0, 86.5) mm | (-34, 0, 88) mm |
| `rh_thbase` | (34, -8.58, 29) mm, 45° about y | (34, -9, 29) mm, 0.785 rad about y |
| Finger ranges | Shadow Robot's: `J3` and `THJ1` from -0.262, `THJ2` ±0.698 | Gym's: `J3` and `THJ1` from 0, `THJ2` ±0.524, the rest rounded to 1e-3 rad |

So the little finger from its knuckle on sits about 1.8 mm and 0.14° away, and every thumb link
0.42 mm away. The asset bundle's `right-openai.xml` is `right.xml` with these changes; its links keep
the default meshes, OpenAI's being the same Shadow Robot meshes, and the metacarpal's stay where they
are in the palm while its hinge moves. Presets are clipped to the narrower ranges. OpenAI only
describes the right hand.

A DexGraspNet pose converts joint by joint: each `robot0:` joint is the next-numbered joint of its
finger (`robot0:FFJ3` is `rh_FFJ4`, ..., `robot0:THJ0` is `rh_THJ1`), negated for `FFJ3`, `MFJ3`,
`THJ1` and `THJ0`; the wrist, `rh_WRJ2` and `rh_WRJ1`, stays at 0. Its palm frame is `rh_palm`'s
with MuJoCo's axes. Converted, every body is where DexGraspNet's own forward kinematics put it,
to float precision.

## API

::: robot_models.hands.shadow.numpy.ShadowHand
