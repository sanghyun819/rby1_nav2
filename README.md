# RBY1 Nav2 Configuration Snapshot

RBY1 로봇의 ROS 2 Humble / Nav2 navigation 설정을 따로 보관하기 위한
configuration snapshot입니다.

이 저장소는 전체 ROS 패키지 소스가 아니라, 운용과 튜닝에 직접 필요한
`bt`, `config`, `launch`, `rby1_nav2` 폴더만 모아둔 백업용 저장소입니다.

## What Is Included

```text
.
├── bt/                 # Behavior Tree XML files
├── config/             # Nav2, AMCL, controller, costmap, planner configs
├── launch/             # Navigation launch files
└── rby1_nav2/          # Python helper/debug nodes
```

## What Is Not Included

대용량 파일과 빌드 산출물은 의도적으로 제외했습니다.

```text
bag/
maps/
build/
install/
log/
__pycache__/
*.pyc
```

## Main Navigation Stack

대표 설정 파일은 `config/test.yaml`입니다.

| Component | Main Setting |
| --- | --- |
| Localization | AMCL, `base_frame_id: base_nav`, `/scan_merged` |
| Controller | MPPI Controller, `motion_model: DiffDrive` |
| Local Costmap | 2D LaserScan + 3D PointCloud obstacle layers |
| Global Costmap | Static map + obstacle layer + inflation |
| Planner | `nav2_smac_planner/SmacPlannerLattice` |
| Lattice | `config/rby1_lattice.json` |
| Recovery | Wait, Clear Costmap, Spin, BackUp, DriveOnHeading |

## Common Launches

기존 ROS workspace 안의 `rby1_nav2` 패키지에 이 폴더들을 복원한 뒤 실행합니다.

```bash
ros2 launch rby1_nav2 navigation.launch.py
```

```bash
ros2 launch rby1_nav2 navigation_restaurant.launch.py
```

맵이나 파라미터 파일을 명시하려면 launch argument로 넘깁니다.

```bash
ros2 launch rby1_nav2 navigation.launch.py \
  map:=/path/to/map.yaml \
  params_file:=/path/to/test.yaml
```

## Key Files

| File | Purpose |
| --- | --- |
| `config/test.yaml` | Main Nav2 parameter file |
| `config/test_stvl.yaml` | STVL/3D obstacle-layer related variant |
| `config/rby1_hri.yaml` | HRI navigation parameter variant |
| `config/rby1_lattice.json` | Smac State Lattice motion primitives |
| `bt/hri_main_bt.xml` | HRI-oriented navigation behavior tree |
| `bt/nav2_escape.xml` | Escape/recovery-oriented behavior tree |
| `rby1_nav2/vfh_plus_escape_node.py` | VFH+ based local escape/debug logic |
| `rby1_nav2/vfh_visualizer_node.py` | VFH visualization/debug node |

## Planner Tuning Notes

현재 planner는 Smac State Lattice를 사용합니다.

```yaml
planner_server:
  ros__parameters:
    GridBased:
      plugin: "nav2_smac_planner/SmacPlannerLattice"
      lattice_filepath: "/home/nvidia/rby1_nav2/src/rby1_nav2/config/rby1_lattice.json"
```

좁은 틈에서 시작할 때 제자리 회전 경로가 만들어질 수 있으므로, 다음 항목을
함께 확인하는 것이 좋습니다.

- `global_costmap` / `local_costmap`의 `footprint`
- `rby1_lattice.json`의 `num_of_headings`
- Smac planner의 `rotation_penalty`
- RViz의 `/global_costmap/published_footprint`
- RViz의 `/plan` 시작 구간 heading 변화

heading bin을 더 촘촘하게 만들면 회전 경로의 discrete collision check가
더 자주 일어나므로 좁은 공간에서 안전성이 좋아질 수 있습니다.

## Restore Workflow

이 저장소 내용을 기존 ROS 패키지 위치로 복사합니다.

```bash
cp -r bt config launch rby1_nav2 /home/nvidia/rby1_nav2/src/rby1_nav2/
```

필요하면 workspace를 다시 빌드합니다.

```bash
cd /home/nvidia/rby1_nav2
colcon build --symlink-install --packages-select rby1_nav2
source install/setup.bash
```

## Notes

- `base_nav`는 navigation 기준 프레임으로 사용됩니다.
- 실제 footprint와 RViz의 published footprint가 겹치는지 항상 확인해야 합니다.
- 대용량 bag/database 파일은 이 저장소에 추가하지 않습니다.
- 운영 환경에서는 map 파일과 launch argument 경로가 로컬 시스템에 맞게 존재해야 합니다.
