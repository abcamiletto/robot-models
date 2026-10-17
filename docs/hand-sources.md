# Hand sources

Each public hand has one canonical description and uses the same shared hand/FK
implementation. Models come from manufacturers, MuJoCo Menagerie, or established
ROS descriptions, with the exact upstream revision and conversion changes recorded.

| Model | Default description | Pinned source | Asset terms |
| --- | --- | --- | --- |
| `ability` | PSYONIC Ability Hand, large | [Manufacturer API](https://github.com/psyonicinc/ability-hand-api/tree/34c9a9324d3739d976e6de441e56ccefafd0000b) | MIT |
| `allegro` | Allegro V3, both sides | [Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/f054586a8e90465d49ee5be15335c4a0c7f57caf/wonik_allegro) | BSD-2-Clause |
| `barrett` | Barrett BH280 | [Robotnik ROS description](https://github.com/RobotnikAutomation/barrett_hand_common/tree/b3d96febf8ef7b2aae4dab6015928918324c364a) | BSD-3-Clause |
| `brainco` | BrainCo Revo 2, both sides | Official SDK `Revo2_xml.zip`, dated 2025-11-05; [manufacturer description](https://github.com/BrainCoTech/revo2_description/tree/92cc697c7fa691db59404ce52344f3969a5ef7a6) | Hosted notice records Apache-2.0 and the SDK archive's coverage limitation |
| `ezgripper` | SAKE EZGripper MuJoCo model | [SAKE-maintained model](https://github.com/vikashplus/ezgripper_sim/tree/0fdf9dbdb5023e9345f392a381cef80b1ebe965d), linked by [SAKE support](https://sakerobotics.com/support/) | Apache-2.0 |
| `inspire` | Inspire RH56 DFQ, both sides | [Unitree G1 integration](https://github.com/unitreerobotics/unitree_ros/tree/5994d4faef0a9cadd3287f8de0199a67eeb2a259/robots/g1_description) | BSD-3-Clause |
| `leap` | LEAP Hand v1, both sides | [Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/f054586a8e90465d49ee5be15335c4a0c7f57caf/leap_hand) | MIT |
| `robotiq-3finger` | Robotiq 3-Finger Adaptive Gripper | [Historical ROS package](https://github.com/ros-industrial-attic/robotiq/tree/45196f6558fe8ba9d89bc8a105396c68c3e7e892/robotiq_3f_gripper_visualization), authored by Robotiq | BSD-2-Clause |
| `shadow` | Shadow Hand E3M5, both sides | [Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/f054586a8e90465d49ee5be15335c4a0c7f57caf/shadow_hand), original Shadow Robot assets | Apache-2.0 |
| `sharpa` | Sharpa Wave, both sides | [Manufacturer description](https://github.com/sharpa-robotics/sharpa-urdf-usd-xml/tree/0d19cac602f46456b819e4b6a2c09a74982c9a3e) | Apache-2.0 |

Robotnik is a robotics integrator; its Barrett description is not represented as
a Barrett Technology release. The Robotiq ROS repository is now archived. SAKE
links the EZGripper simulator on its own support page. Its original model is by
Vikash Kumar; the pinned revision is maintained by SAKE Robotics. These qualifications are part of the provenance.

Existing visual meshes, joint conventions and default API behavior are preserved.
Their hub `LICENSE.md` files describe conversions and license scope. Updated
Barrett, Robotiq and Shadow bundles retain complete upstream license texts,
original descriptions and meshes, provenance and file hashes. Shadow restores
the source forearm collider; see [collision geometry](collisions.md).

```python
from robot_models import create_model

hand = create_model("barrett")
params = hand.get_rest_pose(hands="rest")
links = hand.forward_links(**params)
```

All hands use the same `model_path`, `side` and `hand_pose` conventions. Available
sides follow the actual supplied assets; no mirrored assembly is synthesized.
Barrett, Robotiq-3F and EZGripper each provide one assembly with the API side tag
`right`. All models use robot-models' Y-up frame and floating-root convention.

Joint coordinates describe kinematics rather than motors. Barrett's URDF has
eight hinges and four independent coordinates after four declared mimic
relationships. EZGripper has four hinge coordinates; its native tendon, contact
and actuator definitions remain in the source/runtime XML, but FK does not solve
tendon dynamics. Rest transforms, axes/signs, limits, geometry and declared URDF
mimics follow the recorded source. This validates the software description, rather
than certifying physical calibration or simulation dynamics.

## Rebuild and validate the new canonical bundles

Clone the three repositories above into `UPSTREAMS/barrett`, `UPSTREAMS/robotiq`
and `UPSTREAMS/ezgripper-sim`, and check out their pinned revisions.

```bash
uv run --group asset-preparation python -m scripts.prepare_hand_assets UPSTREAMS results/canonical-hands
uv run --group asset-preparation python -m scripts.check_hand_fidelity \
  results/canonical-hands results/canonical-reference
```

Barrett and Robotiq use the upstream standalone xacros expanded by `xacro`; their
original files remain unchanged. Robotiq's old checked-in generated URDF points
to missing visual STLs, so the current upstream xacro with its original DAE
visuals is used. No substitute mesh is introduced. All visual elements are
retained, including fixed Barrett tactile frames. Runtime meshes are converted
to STL; original URDF collision elements are also retained in the runtime,
including dedicated collision STLs and analytic boxes. Source dynamics remain
available unchanged.
Their runtime inertias are placeholders inferred from visual mesh volume at
density 1000 kg/m³, rather than the source URDF inertials.

EZGripper retains its native MJCF tendons, contacts and actuator. Preparation
removes the scene floor/light, resets the mount position/quaternion, adds a free
root and rewrites mesh paths. Original copyright comments are retained and STL
meshes are copied unchanged. The original MJCF remains under `source/`.

The independent oracle imports the expanded upstream URDF into MuJoCo, without
using robot-models' FK or runtime XML. Only geometry is removed from that FK
reference, with inertia compilation bounds/balancing; these do not affect the
kinematics. Surfaces are compared separately with the original visual and collision meshes.
EZGripper is compared directly with its untouched upstream MJCF. Original joint
ordering, axes, limits, mimic relationships, rest and articulated transforms and
representative surface points are checked with an explicit basis/root mapping.

| Default source | Fixed poses | Largest FK matrix error | Largest surface error (m) |
| --- | ---: | ---: | ---: |
| Barrett / Robotnik | 12 | 1.34e-7 | 3.06e-10 |
| Robotiq-3F / ROS | 26 | 1.80e-7 | 1.23e-9 |
| EZGripper / SAKE-linked MuJoCo | 12 | 1.05e-7 | 2.53e-9 |

Frozen reference fixtures and their upstream licenses are in
`tests/reference_hands/`; NumPy checks these independent source references.
The shared backend consistency, compilation and gradient tests cover PyTorch
and JAX.

## Published bundles

Barrett, Robotiq and Shadow collision updates are mirrored at
[revision `2ae41639`](https://huggingface.co/abcamiletto/robot-models/commit/2ae41639743b4508c58a60250a95dc0b601a88d3), with complete upstream
terms and original source files. All 25 other hub files remain unchanged,
including EZGripper and the other model bundles. Preparation is byte-for-byte
reproducible; archive SHA-256 digests are:

| Bundle | SHA-256 |
| --- | --- |
| `barrett/assets.zip` | `ba4a04cdc5f0045975c6ca61b1e55995bd22bd464d96fdc3721308c5b9fcc051` |
| `robotiq-3finger/assets.zip` | `932b6b9d05b4534ffe0a73a11ec6795ca99fc466596aa6436f54e2981cb6d658` |
| `shadow/assets.zip` | `9ded526a4db0bd679174a87f43e71bacb4eb911196701d33949378689ef9330d` |
| `ezgripper/assets.zip` (unchanged) | `1d3869c8c6f3ed475368abfabe14765a1abf15003e1cdae1bebf363926387ecb` |
