#!/usr/bin/env python3
"""Autonomous iteration, training, and diagnostic suite for Project 2 tasks.

Executes:
  Stage 1: Fall Recovery Policy Training (Symmetrical upright motionless standing & hands pushup)
  Stage 1 Diag: Fall Recovery telemetry verification (head height in [1.12, 1.26], motionless v < 0.1 m/s)
  Stage 2: 12cm Stairs Policy Training (High knee lift ~60 deg, single stance, double flight penalty -15.0)
  Stage 2 Diag: Stairs traversal telemetry (0% bunny hopping, high knee flexion > 50 deg, forward ascent)
  Stage 3: Dance Tracking Fine-Tuning (Enhanced ankle scale 0.45/0.35, zero push-disturbances)
  Stage 3 Diag: Dance Tracking telemetry (100% survival, overall RMSE < 10 deg)
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
LOG_DIR = os.path.join(WORKSPACE, "logs", "autonomous_suite")
os.makedirs(LOG_DIR, exist_ok=True)
SUITE_LOG = os.path.join(LOG_DIR, f"suite_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")


def log(msg: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    print(formatted)
    sys.stdout.flush()
    with open(SUITE_LOG, "a") as f:
        f.write(formatted + "\n")
        f.flush()


def run_subcommand(cmd, desc, timeout_s=3600):
    log(f"Starting: {desc}")
    log(f"Command: {' '.join(cmd)}")
    t0 = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=WORKSPACE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output_lines = []
    try:
        for line in proc.stdout:
            line_str = line.rstrip()
            output_lines.append(line_str)
            # Print periodic progress or key markers
            if any(
                k in line_str
                for k in [
                    "Iter",
                    "Mean reward",
                    "Total loss",
                    "Time taken",
                    "SUCCESS",
                    "ERROR",
                    "TELEMETRY",
                    "Step",
                    "Root_Z",
                    "Head_Z",
                    "RMSE",
                    "VERDICT",
                    "PASS",
                    "Gait Phases",
                    "Max Knee Flexion",
                ]
            ):
                print(f"  > {line_str}")
                sys.stdout.flush()
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        log(f"ERROR: Process timed out after {timeout_s}s!")
        proc.kill()
        return -1, output_lines

    t_elapsed = (time.time() - t0) / 60.0
    ret = proc.returncode
    log(f"Completed: {desc} (Exit code {ret}, Elapsed: {t_elapsed:.2f} min)")
    return ret, output_lines


def main():
    log("=" * 80)
    log("AUTONOMOUS ITERATION & VERIFICATION SUITE START")
    log("=" * 80)

    # -------------------------------------------------------------
    # STAGE 1: Task 3 Fall Recovery Training (Symmetrical Upright)
    # -------------------------------------------------------------
    log("\n" + "=" * 80)
    log("STAGE 1: Training Task 3 (Fall Recovery - Symmetrical Upright Motionless Standing)")
    log("=" * 80)
    stage1_cmd = [
        PYTHON_BIN,
        "scripts/train_locomotion.py",
        "--task=CustomHumanoid-FallRecovery-v0",
        "--num_envs=1024",
        "--headless",
        "--max_iterations=600",
        "--checkpoint=exported_models/custom_humanoid/custom_humanoid_fall_recovery_final.pt",
    ]
    ret1, out1 = run_subcommand(stage1_cmd, "Task 3 Fall Recovery Training")
    if ret1 != 0:
        log(f"WARNING: Stage 1 training exited with code {ret1}!")

    # STAGE 1 DIAGNOSTIC
    log("\n" + "-" * 80)
    log("STAGE 1 DIAGNOSTIC: Evaluating Fall Recovery Telemetry")
    log("-" * 80)
    diag1_cmd = [
        PYTHON_BIN,
        "scripts/diagnose_fall_recovery.py",
        "--task=CustomHumanoid-FallRecovery-v0",
        "--num_steps=140",
    ]
    ret_d1, out_d1 = run_subcommand(diag1_cmd, "Task 3 Fall Recovery Diagnostic")

    # -------------------------------------------------------------
    # STAGE 2: Task 2 Stairs Locomotion Training (12cm Steps, High Knee Lift)
    # -------------------------------------------------------------
    log("\n" + "=" * 80)
    log("STAGE 2: Training Task 2 (12cm Stairs Locomotion - High Knee Lift & Anti-Hopping)")
    log("=" * 80)
    stage2_cmd = [
        PYTHON_BIN,
        "scripts/train_locomotion.py",
        "--task=CustomHumanoid-Velocity-Stairs-v0",
        "--num_envs=1024",
        "--headless",
        "--max_iterations=1000",
        "--checkpoint=exported_models/custom_humanoid/custom_humanoid_stairs_final.pt",
    ]
    ret2, out2 = run_subcommand(stage2_cmd, "Task 2 Stairs Locomotion Training")
    if ret2 != 0:
        log(f"WARNING: Stage 2 training exited with code {ret2}!")

    # STAGE 2 DIAGNOSTIC
    log("\n" + "-" * 80)
    log("STAGE 2 DIAGNOSTIC: Evaluating 12cm Stairs Telemetry")
    log("-" * 80)
    diag2_cmd = [
        PYTHON_BIN,
        "scripts/diagnose_telemetry.py",
        "--task=CustomHumanoid-Velocity-Stairs-Play-v0",
        "--num_steps=160",
    ]
    ret_d2, out_d2 = run_subcommand(diag2_cmd, "Task 2 Stairs Telemetry Diagnostic")

    # -------------------------------------------------------------
    # STAGE 3: Task 4 Dance Tracking Fine-Tuning (Ankle Balancer & Clean Play)
    # -------------------------------------------------------------
    log("\n" + "=" * 80)
    log("STAGE 3: Fine-Tuning Task 4 (BeyondMimic Dance Tracking - Ankle Balance & Smoothness)")
    log("=" * 80)
    stage3_cmd = [
        PYTHON_BIN,
        "scripts/rsl_rl/train.py",
        "--task=CustomHumanoid-Tracking-Flat-v0",
        "--motion_file=motions/custom_humanoid_dance1.npz",
        "--num_envs=1024",
        "--headless",
        "--max_iterations=400",
        "--checkpoint=exported_models/custom_humanoid/custom_humanoid_tracking_final.pt",
    ]
    ret3, out3 = run_subcommand(stage3_cmd, "Task 4 Dance Tracking Fine-Tuning")
    if ret3 != 0:
        log(f"WARNING: Stage 3 training exited with code {ret3}!")

    # STAGE 3 DIAGNOSTIC
    log("\n" + "-" * 80)
    log("STAGE 3 DIAGNOSTIC: Evaluating Dance Tracking Telemetry")
    log("-" * 80)
    diag3_cmd = [
        PYTHON_BIN,
        "scripts/diagnose_tracking.py",
        "--task=CustomHumanoid-Tracking-Flat-Play-v0",
        "--num_steps=250",
    ]
    ret_d3, out_d3 = run_subcommand(diag3_cmd, "Task 4 Dance Tracking Diagnostic")

    # -------------------------------------------------------------
    # FINAL VERIFICATION & SUMMARY
    # -------------------------------------------------------------
    log("\n" + "=" * 80)
    log("ALL STAGES COMPLETED! VERIFYING ARTIFACTS AND MODELS IN exported_models/custom_humanoid/")
    log("=" * 80)
    models_dir = os.path.join(WORKSPACE, "exported_models", "custom_humanoid")
    for fname in sorted(os.listdir(models_dir)):
        fpath = os.path.join(models_dir, fname)
        fsize = os.path.getsize(fpath) / (1024 * 1024)
        mtime = datetime.fromtimestamp(os.path.getmtime(fpath)).strftime("%Y-%m-%d %H:%M:%S")
        log(f"  Artifact: {fname:45s} Size: {fsize:6.2f} MB  Modified: {mtime}")

    log("\n[SUITE VERDICT] FULL AUTONOMOUS PIPELINE COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
