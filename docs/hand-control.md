# Hand control

All hands expose `dofs`, `effort_limits` and `closing` in `dof_names` order:

```python
from robot_models.hands.shadow.numpy import ShadowHand

hand = ShadowHand(side="left")
for name, dof, torque, closing in zip(hand.dof_names, hand.dofs, hand.effort_limits, hand.closing):
    print(name, dof.finger, dof.role, torque, closing)

params = hand.get_rest_pose(preset="open", batch_dims=(2,))
params["hand_pose"] = hand.close(params["hand_pose"], 0.1)
```

`Dof.finger` identifies a finger, palm or wrist. `role` is `flexion`, `abduction`
or `rotation` (including thumb opposition). `closing` is 1 for the coordinates that
close the hand, a finger's or the palm's flexion, and 0 for the others, the wrist's
included. `close` adds radians along `closing` and clamps them to `dof_limits`; other
coordinates stay unchanged. A scalar angle applies to all poses; an array has the pose's batch
shape. Negative angles open. NumPy, Torch and JAX use the same interface.

The `open` preset starts from the flat posture; Robotiq's distal joints use
−0.52 radians to straighten the tips against their source mesh rest rotation.
`closed` uses flexion limits and keeps the other coordinates; MANO's, whose limits reach
beyond any real hand, bends to a natural hand's maximum instead. These are kinematic
poses; they do not check contacts. Fingertip distance need not decrease through an
entire joint range.

Gripper finger labels follow these conventions: Barrett finger 3 is `thumb`,
1 is `index`, 2 is `middle`; Robotiq's middle finger is `thumb`, 1 is `index`,
2 is `middle`; EZGripper F1 is `thumb` and F2 is `index`.

## Torque sources

`effort_limits` holds each coordinate's peak torque in N·m: its joint's
`actuatorfrcrange` in the hand's MJCF (MANO's, built locally, are in code),
which `add_hand` applies to every actuator on that joint. A coupled
coordinate's also drives the hinges it couples. Without a published joint
rating, a fingertip force *F* gives *F* times the coordinate's fingertip lever,
|∂p/∂q| at the open pose. These are best estimates, not calibrations, and
peaks: not continuous holding capacity.

| Hand | Limit per coordinate | Source |
| --- | --- | --- |
| Ability | 1.2 fingers and thumb flexor, 0.3 thumb rotator | One motor through the [ICD](https://github.com/psyonicinc/ability-hand-api/blob/34c9a9324d3739d976e6de441e56ccefafd0000b/Documentation/ABILITY-HAND-ICD.pdf)'s 649:1 and 162.45:1 gears, ~9 N at a fingertip (published pinch force). |
| Allegro V3 | 0.70 | [Wonik's maximum torque](https://www.allegrohand.com/sub/product/p.php?idx=1), all 16 joints. |
| Barrett BH280 | 2.1 fingers, 2.5 spread | 15 N active fingertip force. |
| BrainCo Revo 2 | 0.5 thumb abduction, 1.1 thumb flexion, 2.0 fingers | [Manufacturer description](https://github.com/BrainCoTech/revo2_description/tree/92cc697c7fa691db59404ce52344f3969a5ef7a6) joint efforts. |
| EZGripper | 3.9 palm knuckle, 2.1 knuckle tip | 35 N grasp force ([SAKE brochure](https://sakerobotics.com/?wpdmdl=543)). |
| Inspire RH56 | 1.5 thumb yaw, 3.6 thumb pitch, 1.1–1.3 fingers | 15 N thumb and 10 N fingertip forces ([Inspire](https://en.inspire-robots.com/product/rh56dfx)). |
| LEAP V1 | 0.93 | [ROBOTIS](https://emanual.robotis.com/docs/en/dxl/x/xc330-m288/#specifications) XC330-M288-T stall torque at 5 V, one servo per joint. |
| MANO | 0.4–3.0 bend, 0.4–2.3 spread, 0.2–1.5 twist | Human maximal voluntary torques: fingertip strength (40–50 N for the index) times each joint's lever, weaker for the ring and pinky; a finger's base spread matches its bend ([Li et al. 2003](https://doi.org/10.1016/s0268-0033(03)00178-5)). Hinges no muscle turns hold half their joint's bend. |
| Robotiq 3-Finger | 8.0, 4.6, 2.3 finger joints, 8.6 scissor | 60 N maximum finger force ([manual](https://assets.robotiq.com/website-assets/support_documents/document/3-Finger_PDF_20210617.pdf)). |
| Shadow E3M5 | 0.72–0.9 fingers, 0.81–2.37 thumb, 2.2–4.8 wrist | OpenAI Gym's actuator force ranges; its tendon's on both coupled J2 and J1. |
| Sharpa Wave | 0.19–3.3 | [Manufacturer description](https://github.com/sharpa-robotics/sharpa-urdf-usd-xml/tree/0d19cac602f46456b819e4b6a2c09a74982c9a3e) joint efforts. |
