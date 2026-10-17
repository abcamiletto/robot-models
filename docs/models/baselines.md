# Original grasp baseline hands

These sources preserve the hands used by the pinned original implementations.
Existing `shadow`, `allegro`, other public models, and their defaults are unchanged.
All support NumPy, PyTorch and JAX through `create_model`. Shadow uses
`robot_models.shadow.{numpy,torch,jax}.ShadowHand` with an explicit `source`;
the three gripper classes live in `robot_models.baselines.{numpy,torch,jax}`.

| Factory identifier | Backend class | Coordinates | Skeleton bodies | Source side / root |
| --- | --- | ---: | ---: | --- |
| `barrett-gendexgrasp` | `BarrettGenDexGrasp` | 8 | 10 | right / `bh_base_link` |
| `robotiq-3finger-gendexgrasp` | `Robotiq3FingerGenDexGrasp` | 11 | 16 | right / `palm_base` |
| `ezgripper-gendexgrasp` | `EZGripperGenDexGrasp` | 4 | 7 | left assembly / `world` |
| `shadow`, `source="gendexgrasp"` | `ShadowHand` | 24 | 30 | right / `forearm` |
| `shadow`, `source="dexgraspnet"` | `ShadowHand` | 22 | 45 | right / `robot0:palm` |

Only the source side is accepted; these classes do not synthesize mirrored models.
EZGripper defaults to `side="left"`; the others default to `side="right"`.
`get_rest_pose()` returns source zero angles. `hands="flat"` and `hands="rest"`
return zero clipped to source limits; these are not optimizer initialization presets.
The three grippers expose only the common wrist mapping. Shadow maps all common
anatomical finger joints to source-native bodies; see [one-family source selection](shadow.md).

## Asset availability

The `shadow-dexgraspnet` source bundle is published in
[abcamiletto/robot-models on Hugging Face](https://huggingface.co/abcamiletto/robot-models/tree/main/shadow-dexgraspnet).
It downloads automatically on first use, or explicitly with:

```bash
robot-models download shadow --source dexgraspnet
```

It contains the original MJCF, meshes, contact and penetration annotations, an
unchanged upstream README and inherited license notices, full CC BY-NC 4.0 terms,
the derived runtime files and a provenance/hash manifest. Project-specific content
remains noncommercial; inherited assets retain their original notices.

The four GenDexGrasp runtime bundles and its source-only Allegro bundle still
require local preparation. Their complete redistribution terms are unresolved;
see the licensing record below. `robot-models download all` includes published
models only. Explicit requests for unpublished bundles explain local preparation.

## Prepare local assets

Preparation requires a robot-models repository checkout; the preparation scripts
are not included in the installed wheel. Use these original sources:

- [GenDexGrasp](https://github.com/tengyu-liu/GenDexGrasp), commit
  `29fe7efc558b8cc3ce2d8e67c9c22b014adc21c7`.
- [DexGraspNet](https://github.com/PKU-EPIC/DexGraspNet), commit
  `bd1b13d7248729af117e1d46aaa6266b147a3c7b`.
- GenDexGrasp's original [data.zip](https://drive.google.com/file/d/1WRV7m9AAfDOFE6Z9InIRJwhhlRUSQzCX/view),
  SHA-256 `c8777d2195da312281f51c37cdadd8a46bb1b408aada2e9ffd425bd237357f4e`,
  extracted so `data/urdf/urdf_assets_meta.json` exists under `GENDEX_ASSETS`.

```bash
uv run python scripts/prepare_baseline_assets.py \
  --gendexgrasp "$GENDEX_CHECKOUT" \
  --gendex-archive "$GENDEX_ARCHIVE" \
  --dexgraspnet "$DEX_CHECKOUT" \
  --output results/prepared-baseline-hands

robot-models set barrett-gendexgrasp \
  results/prepared-baseline-hands/barrett-gendexgrasp/assets
```

The preparation script checks pinned checkouts for tracked changes and verifies
the archive digest, then extracts the original GenDexGrasp files directly from
that verified archive. It writes
`<identifier>/assets.zip`, `<identifier>/LICENSE.md` and an extracted `assets/`
directory, following the existing hub layout. Shadow bundle identifiers are asset
folder names, not additional public robot families. Only committed DexGraspNet MJCF files
are copied, so untracked or ignored files cannot enter the source bundle. It never
uploads assets. Archive entry order, timestamps and permissions are fixed for reproducible packaging in
the locked environment. Use a new output directory for each preparation.

Each runtime bundle contains `left.xml` or `right.xml`, display STLs, `source.json`,
`LICENSE.md` and an untouched `source/` tree. The manifest records source hashes,
revision, joint order, limits, native mimic declarations, frame aliases, and changes.
The additional `allegro-gendexgrasp` directory is **source-only**; it has no runtime
XML and is not a second Allegro model.

## Load models and native source files

```python
from pathlib import Path
from robot_models import create_model
from robot_models.baselines import source_assets

path = Path("results/prepared-baseline-hands/shadow-dexgraspnet/assets")
hand = create_model("shadow", source="dexgraspnet", runtime="torch", model_path=path)
params = hand.get_rest_pose()
links = hand.forward_links(**params)

native = source_assets("shadow", source="dexgraspnet", model_path=path)
native.verify()  # SHA-256 check of original files and recorded license texts
mjcf_path = native.description
mesh_path = native.mesh_dir
contacts_path = native.annotations["contacts"]
penetration_path = native.annotations["penetration"]
```

`model_path`, configured paths, and the ordinary asset cache resolve these bundles.
For the source-only Allegro bundle, pass an explicit `model_path` to `source_assets`.
`native.root` is the original working-directory root: GenDexGrasp expects
`data/urdf/...`, while DexGraspNet expects `mjcf/...`. The source descriptions retain
upstream package/relative mesh references. Pass `native.mesh_dir` to the original
hand loader; do not assume that a generic URDF resolver understands GenDexGrasp's
filename rules. DexGraspNet's string-based MJCF parser also needs its original
working directory for its compiler mesh paths.

The original entry points are:

- Barrett: `data/urdf/barrett_adagrasp/model.urdf`.
- Robotiq: `data/urdf/robotiq_3finger/urdf/robotiq_3finger_description.urdf`.
- EZGripper: `data/urdf/ezgripper/ezgripper.urdf`.
- GenDexGrasp Shadow: `data/urdf/sr_grasp_description/urdf/shadowhand.urdf`.
- DexGraspNet Shadow: `mjcf/shadow_hand_wrist_free.xml`.
- Allegro source: `data/urdf/allegro_hand_description/allegro_hand_description_left.urdf`.

GenDexGrasp's `annotations["contacts"]` holds **original mesh vertex-index bases**.
Runtime STL deduplication changes vertex indexing: use the original meshes and
original `HandModel` to resolve those indices. The source bundle retains the
original area/volume sampling inputs and loader conventions. Runtime meshes do
not implement the original sampling algorithm or contact allocation.

DexGraspNet retains `contact_points.json` (140 palmar candidates),
`penetration_points.json` (19 keypoints), all original meshes, and the MJCF shapes.
Its original algorithm uses analytic capsule distances and the supplied bevelled
`box.obj` mesh for box signed distance. Preserve its collision-link exclusions,
sampling and self-penetration algorithm by using those native files. Runtime
meshes are for rendering; substituting their convex hulls or uniform samples would
change the baseline again. Display capsules explicitly normalize Trimesh's
centering; this avoids the version-dependent display shift in the historical
loader, without changing native source bytes or analytic distances.

Generated MJCFs are kinematic/display descriptions. Their compiler mass/inertia
bounds provide valid moving bodies for MuJoCo loading; they do not reproduce
source dynamics. Original descriptions retain their supplied dynamics/collision
metadata separately.

## Coordinate and joint mapping

For all five new variants, pose angles map **identically by original name, sign,
and depth-first joint order**. Read `hand.actuated_joint_names` or the manifest;
do not shift canonical Shadow names. No canonical wrist alignment is involved.
Barrett and EZGripper retain independent coordinates even where the URDF declares
mimics: the actual GenDexGrasp vendored kinematics ignores mimic coupling. The
unmodified URDFs and manifest retain those declarations for other consumers.

For original link transform `T_source(q)` and body-local point `p_source`, let:

```text
C = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
B = diag(C, 1)
T_model(q) = B @ T_source(q) @ B.T
p_model   = C @ p_source
```

The native source root is identity for these definitions. Nonidentity child rest
transforms, including the wrist rest rotation, remain inside `T_source` and are
preserved, not stripped by basis conversion. Global translation maps as `C @ t`;
global rotation maps as `C @ R @ C.T`. GenDexGrasp additionally applies `Rx(pi/2)`
to Shadow mesh coordinates **after mesh scale and before visual origin**. This
loader behavior is present in the derived meshes, although absent from its URDF.

DexGraspNet's vendored parser represents each hinge with a fixed pre-joint link
and a moving `*_child` link. The runtime skeleton retains both (45 bodies for 22
coordinates) so annotation frame names match directly. EZGripper's source root
`world` is called `ezgripper_world` in runtime MJCF, because MuJoCo reserves
`world` for its own world body. This sole alias is recorded in `source_link_to_model`;
it has identity rest transform and does not affect joints or annotations.

Use existing `create_model("allegro", side="left")` for GenDexGrasp Allegro.
Original `joint_<i>.0` maps with positive sign to
`("ff", "mf", "rf", "th")[i // 4] + "j" + str(i % 4)`.
Original vector order is not numeric joint order: pack by name. Link frames need
constant rest transforms as well as palm alignment; the audit derives and records
these matrices per articulated body. Limits match within `5.4e-8` rad. No Allegro
geometry or default side was changed.

## Independent fidelity checks

The audit executes the actual upstream `HandModel` class definitions and vendored
kinematics. AST extraction only avoids visualization/optional accelerator imports;
class and rotation-function bodies are unchanged. DexGraspNet uses zero sampled
surface points, so PyTorch3D is unnecessary; face indexing is ordinary tensor
indexing. The audit runs each vendored package in a separate process and explicitly
imports this checkout's `src/robot_models`, overriding installed older versions.

```bash
uv sync --frozen --group dev --group baseline-audit --all-extras

uv run --group baseline-audit python scripts/check_baseline_fidelity.py \
  --project gendexgrasp --upstream "$GENDEX_CHECKOUT" \
  --original-assets "$GENDEX_ASSETS" \
  --bundles results/prepared-baseline-hands --output results/fidelity \
  --write-fixtures results/fidelity-fixtures

uv run --group baseline-audit python scripts/check_baseline_fidelity.py \
  --project dexgraspnet --upstream "$DEX_CHECKOUT" \
  --bundles results/prepared-baseline-hands --output results/fidelity \
  --write-fixtures results/fidelity-fixtures

uv run --group baseline-audit python scripts/check_baseline_fidelity.py \
  --project allegro --upstream "$GENDEX_CHECKOUT" \
  --original-assets "$GENDEX_ASSETS" \
  --bundles results/prepared-baseline-hands --allegro-assets "$LEFT_ALLEGRO_ASSETS" \
  --output results/fidelity
```

The CLI fails if reference inputs are unavailable or any comparison fails. It
checks zero, clipped zero, both limits, two interior poses, and a positive
perturbation of **each** original joint. It compares all link transforms, exact
coordinate ordering and limits, every source mesh vertex and triangle for
GenDexGrasp and DexGraspNet boxes, representative triangle interior points, all
resolved GenDexGrasp contact bases, and all DexGraspNet annotation frames. Every
display capsule vertex is checked against its original analytic surface.
Allegro checks every articulated body with explicit constant rest-frame mappings,
and full palm-relative surfaces at rest and two articulated poses against an
independent sampling-noise control.

Local numerical fixtures can be tested in NumPy, Torch and JAX with
`uv run pytest tests/test_baselines.py`. Configure the five prepared asset paths
first, or place the extracted directories under `tests/assets/models_hub/<identifier>`.
`ROBOT_MODELS_BASELINE_FIXTURES` overrides the default `results/fidelity-fixtures`
under the repository root.
Source/geometry fixtures stay local while licensing is unresolved. Ordinary CI
runs synthetic conversion/frame/mimic regressions; optional source fixture and
cross-model tests explain when local assets are absent.

See [the validation record](baseline-validation.md) for measured errors. These
checks establish model fidelity; broader dataset and Isaac Gym grasp success-rate
reproduction is a separate task.

## Licenses and distribution

The preparation output retains original files and package metadata, includes an
asset-specific `LICENSE.md`, and records changes. No upstream assets are embedded
in the Python package or committed in this change.

- The supplied GenDexGrasp `sr_grasp_description/package.xml` declares `Private`.
  Its complete source directory has no license text that resolves this declaration.
- EZGripper's `package.xml` and Allegro's `manifest.xml` declare BSD, but their
  supplied directories lack the corresponding full license texts. Barrett and
  Robotiq directories also lack license files. The Barrett URDF attributes its source to
  [RobotnikAutomation/barrett_hand_common](https://github.com/RobotnikAutomation/barrett_hand_common/tree/kinetic-devel/barrett_hand_description).
  These observations are recorded without assigning replacement licenses.
- [DexGraspNet's README](https://github.com/PKU-EPIC/DexGraspNet#license) declares
  CC BY-NC 4.0 for its work/dataset. The bundle retains that README, author
  attribution, full CC terms and the unchanged inherited OpenAI notice. Its
  Apache-2.0 grant identifies Vikash Kumar code; it does not establish a blanket
  Apache license for every referenced ShadowRobot model or mesh.
  Do not relabel the full bundle Apache-2.0: project annotations have separate terms.

### Redistribution review (2026-10-17)

| Source bundle | Evidence | Distribution |
| --- | --- | --- |
| DexGraspNet Shadow | Pinned README grants CC BY-NC 4.0; inherited OpenAI/Shadow notice is retained | Published for noncommercial reuse with attribution and both sets of terms |
| GenDexGrasp Barrett | URDF attributes Robotnik; [Robotnik's original project has BSD-3-Clause terms](https://github.com/RobotnikAutomation/barrett_hand_common/blob/kinetic-devel/LICENSE), but the supplied modified source/contact bundle has no complete grant | Local preparation only |
| GenDexGrasp Robotiq-3F | Supplied directory has no license; a license from another Robotiq description does not establish permissions for this exact description and contact annotations | Local preparation only |
| GenDexGrasp EZGripper | Package declares BSD but includes no full license; the complete modified source/contact bundle has no established redistribution grant | Local preparation only |
| GenDexGrasp Shadow | Supplied package declares `Private`; no resolving license text | Local preparation only |
| GenDexGrasp Allegro source | Manifest declares BSD but includes no full license; original contact annotations have no established grant | Local preparation only; existing left Allegro runtime remains published |

The pinned GenDexGrasp checkout contains no project-level asset redistribution
grant. A public download link and licenses on other projects are not substituted
for permissions on these exact bundles. No GenDexGrasp bundle is uploaded pending
clarification from the applicable rights holders.

Publishing preserves the hub layout `<identifier>/{assets.zip,LICENSE.md}` and
updates the hub model card's per-folder license table. The ZIP includes the same
notice and all original source files; `source.json` records source hashes, upstream
revision, redistribution status and, for DexGraspNet, the scoped license paths.
Existing published model folders and their licenses are preserved.
