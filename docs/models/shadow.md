# Shadow

Shadow has one model family and one `ShadowHand` class per array runtime. Select
the upstream description with `source`; omitting it keeps the existing Menagerie
E3M5 model, coordinates, meshes and presets.

```python
from robot_models.shadow.numpy import ShadowHand
from robot_models import create_model

hand = ShadowHand()  # source="menagerie", unchanged default
native = ShadowHand(source="dexgraspnet")
same_class = create_model("shadow", source="dexgraspnet")
print(ShadowHand.available_sources)
print(native.source, native.num_dofs)  # dexgraspnet, 22
```

`list_models()` lists `shadow` once. The former `shadow-gendexgrasp` and
`shadow-dexgraspnet` factory names/classes are replaced by source selection.
Their existing asset folder and cache/config keys remain internal bundle names.

| Source | Pose coordinates | Skeleton frames | Root | Sides | Assets |
| --- | ---: | ---: | --- | --- | --- |
| `menagerie` (default) | 24 | 25 | Forearm | right, left | Automatic download; Apache-2.0 |
| `gendexgrasp` | 24 | 30 | Forearm | right | Local preparation; redistribution unresolved |
| `dexgraspnet` | 22 | 45 | Palm | right | Automatic download; CC BY-NC 4.0 and inherited notices |

## Physical joints, coordinates and motors

The 24-versus-22 difference comes from including or omitting the two wrist
coordinates. DexGraspNet places the whole palm-rooted hand with a separate global
rotation and translation. Its 45 skeleton frames include fixed pre-joint and
moving post-joint frames used by the source parser and annotations; they do not
represent 45 mechanical joints.

`num_dofs` and `actuated_joint_names` describe the library's independent pose
coordinates, not a count of motors. The checked Menagerie MJCF has 24 hinge
coordinates and 20 actuators: four distal finger pairs share tendon actuation.
The library exposes independent kinematics, rather than simulating those motors.

Sources preserve different rest frames, angle signs/order, limits, geometry and
collision/contact definitions. DexGraspNet uses analytic collision primitives and
hand-picked annotations. These are differences in published descriptions and
algorithms; selecting a source is not a claim of different motor hardware or a
certification that all source calibrations match. `hand_pose` is source-specific;
angles must not be copied between sources by merely matching shifted names.

## Setup

The assets come from MuJoCo Menagerie `shadow_hand` (Apache-2.0). They download on first use from the
[`abcamiletto/robot-models`](https://huggingface.co/abcamiletto/robot-models)
Hugging Face repository, with the upstream license and a list of changes. To prefetch them:

```bash
robot-models download shadow
robot-models download shadow --source dexgraspnet
```

A manual `model_path` is a directory with `left.xml`, `right.xml` and their meshes.
Native baseline sources provide only `right.xml`. Configure each independently:

```bash
robot-models set shadow /path/to/menagerie/assets
robot-models set shadow /path/to/gendexgrasp/assets --source gendexgrasp
robot-models unset shadow --source gendexgrasp
```

`--source menagerie` addresses the same path as the omitted/default source.
`download all` fetches both published Shadow bundles and keeps their paths
separate. The GenDexGrasp bundle requires [local preparation](baselines.md).

## Usage

```python
from robot_models.shadow.numpy import ShadowHand

hand = ShadowHand(side="right")
params = hand.get_rest_pose(hands="rest")
```

## Notes

All sources use the shared hand/FK implementation. `common_joints` maps the same
anatomical roles to each source's native bodies without rewriting their frames.
The Menagerie `flat` and `rest` presets are unchanged. Native baseline sources
return source zero for the default pose and zero clipped to their joint limits
for `flat`/`rest`.

Original baseline URDF/MJCF files and optional annotations remain available:

```python
from robot_models.baselines import source_assets

native = source_assets("shadow", source="dexgraspnet")
native.verify()
original_mjcf = native.description
contacts = native.annotations["contacts"]
penetration = native.annotations["penetration"]
```

See [baseline compatibility and provenance](baselines.md) for the original
algorithms, source-specific conventions and redistribution terms.

## API

::: robot_models.shadow.numpy.ShadowHand
