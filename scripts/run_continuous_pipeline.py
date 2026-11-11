#!/usr/bin/env python3
"""Continuous autonomous pipeline runner for Project 2.

Executes Task 2 (Stairs 2000 iters), Task 3 (Fall Recovery 3000 iters),
and Task 4 (Dance Tracking 3500 iters) sequentially with real-time logging.
"""

import os
import subprocess
import sys
import time
from datetime import datetime

# 优先读环境变量；默认使用当前解释器和 scripts/ 的上一级目录，避免硬编码个人路径
PYTHON_BIN = os.environ.get("PYTHON_BIN", sys.executable)
WORKSPACE = os.environ.get(
    "WBT_WORKSPACE", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

tasks = [
    {
        "name": "Task 2: Stairs Locomotion (Joint Scales & High Leg Lift)",
        "task_name": "CustomHumanoid-Velocity-Stairs-v0",
        "num_envs": 1024,
        "max_iterations": 2000,
    },
    {
        "name": "Task 3: Fall Recovery (Head Height >= 1.05m Upright)",
        "task_name": "CustomHumanoid-FallRecovery-v0",
        "num_envs": 1024,
        "max_iterations": 3000,
    },
    {
        "name": "Task 4: BeyondMimic Dance Tracking (Full Joint Amplitude)",
        "task_name": "CustomHumanoid-Tracking-Flat-v0",
        "num_envs": 1024,
        "max_iterations": 3500,
    },
]

print("=" * 80)
print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] STARTING PROJECT 2 FULL PIPELINE")
print("=" * 80)

for idx, t in enumerate(tasks, 1):
    t_start = time.time()
    print("\n" + "#" * 80)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] STAGE {idx}/{len(tasks)}: {t['name']}")
    print(f"Command: {PYTHON_BIN} scripts/train_locomotion.py --task={t['task_name']} --num_envs={t['num_envs']} --headless --max_iterations={t['max_iterations']}")
    print("#" * 80 + "\n")
    sys.stdout.flush()

    cmd = [
        PYTHON_BIN,
        "scripts/train_locomotion.py",
        f"--task={t['task_name']}",
        f"--num_envs={t['num_envs']}",
        "--headless",
        f"--max_iterations={t['max_iterations']}",
    ]

    ret = subprocess.call(cmd, cwd=WORKSPACE)
    t_elapsed = (time.time() - t_start) / 60.0

    if ret == 0:
        print(f"\n[STAGE {idx} SUCCESS] {t['name']} completed in {t_elapsed:.1f} minutes!")
    else:
        print(f"\n[STAGE {idx} ERROR] Process exited with return code {ret} after {t_elapsed:.1f} minutes!")
        sys.exit(ret)
    sys.stdout.flush()

print("\n" + "=" * 80)
print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ALL STAGES COMPLETED SUCCESSFULLY!")
print("=" * 80)
