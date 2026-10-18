# Hand control

All hands expose `dofs` and `effort_limits` in `actuated_joint_names` order:

```python
from robot_models.hands.shadow.numpy import ShadowHand

hand = ShadowHand(side="left")
for name, dof, torque in zip(hand.actuated_joint_names, hand.dofs, hand.effort_limits):
    print(name, dof.finger, dof.role, dof.closing, torque)

params = hand.get_rest_pose(hands="open", batch_dims=(2,))
params["hand_pose"] = hand.close(params["hand_pose"], 0.1)
```

`Dof.finger` identifies a finger, palm or wrist. `role` is `flexion`, `abduction`
or `rotation` (including thumb opposition). `closing` is the coordinate sign for
closing, or zero. Wrist motion has zero closing. `close` adds radians to closing
flexion coordinates and clamps them to their limits; other coordinates stay
unchanged. A scalar angle applies to all poses; an array has the pose's batch
shape. Negative angles open. NumPy, Torch and JAX use the same interface.

The `open` preset starts from the flat posture; Robotiq's distal joints use their
lower bounds to account for the source rest frames. `closed` uses flexion limits
and keeps the other coordinates. Existing presets are unchanged. These are
kinematic poses; they do not check contacts. Fingertip distance need not decrease
through an entire joint range.

Gripper finger labels follow these conventions: Barrett finger 3 is `thumb`,
1 is `index`, 2 is `middle`; Robotiq's middle finger is `thumb`, 1 is `index`,
2 is `middle`; EZGripper F1 is `thumb` and F2 is `index`.

## Torque sources

`effort_limits` contains peak joint torque magnitudes in N·m. `inf` means no
verified figure for that pose coordinate in the sources below. Grip force,
payload, sensor range and motor current are not converted into joint torque.
Peak ratings do not establish continuous holding capacity or thermal limits.

| Hand | Limit per coordinate | Source and scope |
| --- | --- | --- |
| Ability | `inf` | [PSYONIC specifications](https://www.psyonic.io/ability-hand): no joint-torque figure. |
| Allegro V3 | 0.70 | [Wonik V3 manual, technical specifications, p. 3](https://manuals.plus/m/2cc66525d067944ab5f536be5a0e8d205018ea74bc765a1cc4062f7b74107b42.pdf#page=10): maximum output torque, all 16 joints. |
| Barrett BH280 | `inf` | [Barrett datasheet](https://web.barrett.com/support/BarrettHand_Documentation/BH8-280_Datasheet.pdf): payload and torque-sensor range, not joint drive limits. |
| BrainCo Revo 2 | `inf` | [Manufacturer parameters](https://brainco.tech/docs/revolimb-hand/en/revo2/parameters.html): grip force and current, no joint-torque figures. |
| EZGripper | `inf` | [SAKE specifications](https://sakerobotics.com/): MX-64AR servo and grip force; no torque mapping for the independent tendon-driven coordinates. |
| Inspire RH56 | `inf` | [Manufacturer selection guide](https://en.inspire-robots.com/wp-content/uploads/2023/11/Inspire-Robots-Products-Selection-Guide-1-2.pdf): fingertip force, no joint-torque figures. |
| LEAP V1 | 0.93 | [LEAP parts list](https://v1.leaphand.com/parts) uses 16 XC330-M288-T servos at 5 V; [ROBOTIS](https://emanual.robotis.com/docs/en/dxl/x/xc330-m288/#specifications) gives 0.93 N·m **stall** torque at 5 V. |
| Shadow E3M5 | `inf` | [Shadow technical specification](https://www.shadowrobot.com/wp-content/uploads/2022/03/shadow_dexterous_hand_e_technical_specification.pdf): no per-coordinate N·m limits. Existing MJCF actuator bounds remain active, including coupled tendons. |
| Sharpa Wave | `inf` | [W01 specifications](https://www.sharpa.com/pages/w01): no per-coordinate N·m figures. |
| Robotiq 3-Finger | `inf` | [Manufacturer manual](https://assets.robotiq.com/website-assets/support_documents/document/3-Finger_PDF_20210617.pdf): grasp force, no independent joint-torque limits. |

`add_hand` applies finite limits to existing joint actuators, accounting for gear
ratios and preserving stricter source force bounds. Unknown limits and tendon
actuators retain their source settings. It does not create missing actuators.
