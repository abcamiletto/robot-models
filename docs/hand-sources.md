# Hand sources

Each hand uses one canonical description. Pinned sources and asset licenses are:

| Model | Default description | Pinned source | Asset terms |
| --- | --- | --- | --- |
| `ability` | PSYONIC Ability Hand, large | [Manufacturer API](https://github.com/psyonicinc/ability-hand-api/tree/34c9a9324d3739d976e6de441e56ccefafd0000b) | MIT |
| `allegro` | Allegro V3, both sides | [Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/f054586a8e90465d49ee5be15335c4a0c7f57caf/wonik_allegro) | BSD-2-Clause |
| `barrett` | Barrett BH280 | [Robotnik ROS description](https://github.com/RobotnikAutomation/barrett_hand_common/tree/b3d96febf8ef7b2aae4dab6015928918324c364a) | BSD-3-Clause |
| `brainco` | BrainCo Revo 2, both sides | Official SDK `Revo2_xml.zip`, dated 2025-11-05; [manufacturer description](https://github.com/BrainCoTech/revo2_description/tree/92cc697c7fa691db59404ce52344f3969a5ef7a6) | Hosted notice records Apache-2.0 and the SDK archive's coverage limitation |
| `ezgripper` | SAKE EZGripper MuJoCo model | [SAKE-maintained model](https://github.com/vikashplus/ezgripper_sim/tree/0fdf9dbdb5023e9345f392a381cef80b1ebe965d), linked by [SAKE support](https://sakerobotics.com/support/) | Apache-2.0 |
| `inspire` | Inspire RH56 DFQ, both sides | [Unitree G1 integration](https://github.com/unitreerobotics/unitree_ros/tree/5994d4faef0a9cadd3287f8de0199a67eeb2a259/robots/g1_description) | BSD-3-Clause |
| `leap` | LEAP Hand v1, both sides | [Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/f054586a8e90465d49ee5be15335c4a0c7f57caf/leap_hand) | MIT |
| `mano` | MANO, both sides, built locally | The user's own [MANO](https://mano.is.tue.mpg.de/) files, through Body Models | [MANO license](https://mano.is.tue.mpg.de/license.html): not redistributed |
| `robotiq-3finger` | Robotiq 3-Finger Adaptive Gripper | [Historical ROS package](https://github.com/ros-industrial-attic/robotiq/tree/45196f6558fe8ba9d89bc8a105396c68c3e7e892/robotiq_3f_gripper_visualization), authored by Robotiq | BSD-2-Clause |
| `shadow` | Shadow Hand E3M5, both sides | [Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/f054586a8e90465d49ee5be15335c4a0c7f57caf/shadow_hand), original Shadow Robot assets | Apache-2.0 |
| `sharpa` | Sharpa Wave, both sides | [Manufacturer description](https://github.com/sharpa-robotics/sharpa-urdf-usd-xml/tree/0d19cac602f46456b819e4b6a2c09a74982c9a3e) | Apache-2.0 |


Robotnik is an integrator, not Barrett Technology. The Robotiq ROS repository is
archived. SAKE links the EZGripper model on its support page; Vikash Kumar wrote
the original model and SAKE maintains the pinned revision.

Assets download from [Hugging Face](https://huggingface.co/abcamiletto/robot-models),
except MANO's: they are [built locally](models/mano.md#assets-are-built-locally).
Bundle `LICENSE.md` files record license scope and changes. Barrett, Robotiq and
Shadow bundles also include original descriptions, meshes and file hashes.
Barrett, Robotiq and EZGripper supply one assembly tagged `right`; the other
hands supply both sides. Pose coordinates describe joints, not motor counts.
Shadow's right hand also offers OpenAI Gym's kinematics, in its own MJCF on the
same meshes; see [OpenAI kinematics](models/shadow.md#openai-kinematics).

## Preparation and validation

Barrett and Robotiq use expanded upstream xacros with all visual and collision
elements. Robotiq uses the xacro's original DAE visuals because the checked-in
URDF references missing STLs. Runtime meshes are converted to STL. Their inertias
are inferred from visual volume at 1000 kg/m³, not copied from the URDF.

EZGripper keeps its tendons, contacts and actuator. Preparation removes scene
floor/light, resets the mount, adds a free root and updates mesh paths; STL files
and copyright comments stay intact. FK does not solve tendon dynamics.
Shadow restores the source forearm collider while preserving existing visuals
and joint frames.

Independent MuJoCo comparisons check ordering, axes, limits, couplings, rest and
articulated FK, and original visual/collision surfaces with an explicit frame
mapping. Reference FK uses upstream descriptions without robot-models:

| Default source | Fixed poses | Largest FK matrix error | Largest surface error (m) |
| --- | ---: | ---: | ---: |
| Barrett / Robotnik | 12 | 1.34e-7 | 3.06e-10 |
| Robotiq-3F / ROS | 26 | 1.80e-7 | 1.23e-9 |
| EZGripper / SAKE-linked MuJoCo | 12 | 1.05e-7 | 2.53e-9 |


Shadow was checked at five poses per side: body transforms match exactly and
surface error is below 8e-9 m. Its OpenAI kinematics match DexGraspNet's own forward
kinematics of OpenAI's MJCF ([`bd1b13d`](https://github.com/PKU-EPIC/DexGraspNet/tree/bd1b13d7248729af117e1d46aaa6266b147a3c7b))
at 3,000 random poses, every body within 1e-9 m. Fixtures and licenses remain in
`tests/reference_hands/`. Original preparation and audit tools are in
[publication history](https://github.com/abcamiletto/robot-models/tree/27fdab1/scripts).

## Published bundles

Collision updates are published at Hugging Face
[revision `3d1aa6d0`](https://huggingface.co/abcamiletto/robot-models/commit/3d1aa6d070de82469cc3e87e20d34dbba05a9e89),
Shadow's OpenAI kinematics at
[revision `331c5b93`](https://huggingface.co/abcamiletto/robot-models/commit/331c5b935e3a52215d4e38dcef47c3739095b168),
effort limits at
[revision `9ab6b8eb`](https://huggingface.co/abcamiletto/robot-models/commit/9ab6b8ebed54bd60567ae084aea560ac81f0e1d8).
Archive SHA-256 digests:

| Bundle | SHA-256 |
| --- | --- |
| `barrett/assets.zip` | `1025fe38abe64be9af4fc47de6008cfb7362f99a0c0970bdf6f58f3a525fbbc7` |
| `robotiq-3finger/assets.zip` | `a96b21ba10dc3979a79b8de8ad9c045b45a4625544a12ef71615e05ba893f2b5` |
| `shadow/assets.zip` | `775535052bf968bf6e6a5cce1daf63b8f14348cb458256c7e13b52e166e7472b` |
| `ezgripper/assets.zip` | `25a27e4239677aec5c5f558105625a8404e754eb9ad9e14d0543c1ea1b25d681` |
