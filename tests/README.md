# GenManip 回归测试

所有命令在仓库根目录执行。测试使用 Python 标准库 `unittest`，不需要 pytest。
它们验证局部逻辑，不代表 EBench 任务成功率，不会启动 SimulationApp、机器人或 GPU 仿真，也不会修改场景资产。

## 目录与覆盖范围

| 功能类别 | 测试文件（相对 tests/） | 被测生产代码（相对 genmanip/） | 检查内容 |
| --- | --- | --- | --- |
| Metric：位置范围判定 | `unit/test_range_local_offset.py` | `extensions/metrics/default/sr_based_genmanip_range.py` | 检测点是否在目标范围内：默认使用原点、局部偏移随物体旋转、计算不修改原始位姿、非法偏移被拒绝 |
| Metric：堆叠关系判定 | `unit/test_stack_support.py` | `extensions/metrics/default/sr_based_genmanip_relationship.py` | 默认仍走旧关系逻辑；新模式检查几何支撑、上下顺序、悬空/穿透、容器覆盖及阈值校验 |
| Skill：手臂自定义动作 | `unit/test_object_position_motion.py` | `extensions/skills/default/custom_motion/custom_motion.py` | object_position 的目标点跟随物体而目标朝向保持世界朝向；原 object_frame 仍继承物体旋转；非法参考点被拒绝 |
| Skill：底座移动 | `unit/test_move_to.py` | `extensions/skills/default/move_to/move_to.py` | 默认/慢速步长、累计位移、移动方向、转向行为、零移动及步长参数边界 |
| Demogen：初始场景布局 | `unit/test_custom_tableset.py` | `demogen/random_place/random_place.py` | 物体局部扰动范围、保留高度和朝向、旋转兜底默认开启/显式关闭，以及无可用放置位置时的处理 |
| Demogen：放置稳定性验收 | `unit/test_verify_placement_quaternion.py` | `demogen/random_place/random_place.py` | 静置前后位姿比较：q/-q 等价不误判，真实大旋转和过大位移仍被拒绝 |
| 场景加载：碰撞预处理 | `unit/test_collider_preprocess.py` | `utils/loader/scene.py` | 只处理指定物体；未指定时保持旧全局行为；空列表不处理物体；非法或不存在的 UID 在修改前报错 |
| USD 工具：网格变换解析 | `usd/test_baked_mesh_transform.py` | `utils/usd_utils/transform_utils.py` | 用真实 USD 内存场景验证：空 xformOpOrder 忽略引用层残留变换，普通平移/缩放仍正常解析 |

Metric 测试检查“是否满足成功条件”的局部判定；Skill 测试检查动作目标和移动增量的计算，
不执行真实机械臂规划或底座运动。Demogen 测试检查生成演示前的布局和验收逻辑，
不代表物体在真实仿真中一定稳定。场景加载测试检查调用范围，不验证碰撞体实际 cooking；
USD 测试检查变换解析，不覆盖所有资产或完整点云提取流程。

`support_container_uid` 的自动 ID 映射已撤回，因此不再测试这个不存在的功能；
metric 对固定容器 ID 的覆盖判定仍保留测试。

## 普通 Python 环境：运行 unit

要求 Python 3.10 或更新版本、NumPy、SciPy、Pydantic 2 和 Shapely 2。
建议使用独立虚拟环境，避免更改 Isaac Sim 自带依赖：

```bash
python3 -m venv tests/.venv
tests/.venv/bin/python -m pip install -r tests/requirements.txt
tests/.venv/bin/python -m unittest discover -s tests/unit -t . -v
```

不需要安装仓库根目录的全部 requirements，也不需要 Isaac Sim、cuRobo、Open3D 或 GPU。

## 已有 Isaac Sim 环境

本机 Isaac Sim 位于 `/isaac-sim`；其他机器请替换该路径。先退出已激活的 Conda 环境，
避免 Isaac 启动脚本的环境混用警告。已有环境须具备上述数值计算依赖。

只运行 unit：

```bash
/isaac-sim/python.sh -m unittest discover -s tests/unit -t . -v
```

包含 USD 测试的完整检查：

```bash
PYTHONPATH="/isaac-sim/kit/exts/omni.usd.libs${PYTHONPATH:+:$PYTHONPATH}" \
LD_LIBRARY_PATH="/isaac-sim/kit/exts/omni.usd.libs/bin:/isaac-sim/kit${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  /isaac-sim/python.sh -m unittest discover -s tests -t . -v
```

USD 测试需要能导入 `pxr.Usd`、`pxr.UsdGeom`、`pxr.Gf` 的 OpenUSD Python 环境。
上面的路径适用于当前 Isaac 安装，不保证适用于所有 Isaac 版本。
若普通 Python 已安装兼容的 OpenUSD Python bindings，也可直接使用该 Python 运行完整检查。
缺少 `pxr` 时完整检查会报依赖错误，不会静默跳过；只想运行无 USD 测试时使用 `tests/unit` 命令。

单独运行某个模块或用例：

```bash
/isaac-sim/python.sh -m unittest tests.unit.test_move_to -v
/isaac-sim/python.sh -m unittest tests.unit.test_range_local_offset.RangeLocalOffsetTest.test_default_still_measures_origin -v
```

## 测试边界与维护

部分生产模块在导入时需要 Isaac/cuRobo。unit 测试用 AST 从生产源码提取实际函数或类，
并注入最小依赖替身；测试中没有另写一份待测算法。测试路径根据 `__file__` 定位仓库。
这种方式覆盖算法、分支和参数校验，但不能证明完整模块导入、物理接触、规划器或模型评测正确。
USD 测试则加载真实变换模块，使用内存 stage 检查引用层与变换行为。

修改实现后运行相关模块；提交前运行完整检查。新增功能应同时覆盖默认兼容行为、
新行为和失败边界。不要通过忽略依赖错误或跳过失败来制造全绿结果。
Shapely 的 `vectorized.contains` 弃用警告来自现有生产代码，不是测试失败。

本次验证环境：Python 3.10（Isaac 自带）、NumPy 1.26.4、SciPy 1.15.2、Pydantic 2.12.5、Shapely 2.1.1；完整套件 34 项通过。
独立虚拟环境的全新依赖安装因下载缓慢中止，尚未完成该环境的运行验证。
