#!/usr/bin/env bash
# ============================================================
# scripts/wait_and_eval.sh
# ------------------------------------------------------------
# 轮询 BMN 训练进程, 训练结束后自动:
#   1) finalize manifest.json (status: completed)
#   2) 找出最大 epoch 的 ckpt (BMN_epoch_NNNNN.pdparams)
#   3) 删 cached predictions, 强制重新前向
#   4) 跑 pp eval, 写 metrics_final.json
#   5) 打印一行 final 摘要
#
# 用法 (推荐 setsid detach):
#   setsid bash scripts/wait_and_eval.sh <train_pid> <run_dir> > /tmp/wait_eval.log 2>&1 &
#   disown
#
# 或者只给 train_pid, 自动从 experiments/ 找最近的 run_dir:
#   setsid bash scripts/wait_and_eval.sh <train_pid> > /tmp/wait_eval.log 2>&1 &
#   disown
#
# 退出码:
#   0  成功 (训练完成 + eval 成功)
#   1  用户输入错
#   2  训练进程不存在
#   3  eval 失败
# ============================================================

set -u

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

TRAIN_PID="${1:-}"
RUN_DIR="${2:-}"

if [ -z "$TRAIN_PID" ]; then
    echo "USAGE: $0 <train_pid> [<run_dir>]"
    echo "  e.g. $0 3571466"
    exit 1
fi

# 自动找 run_dir (最近修改的, name 含 bmn_pingpong)
if [ -z "$RUN_DIR" ]; then
    RUN_DIR=$(ls -dt experiments/*-train-bmn_pingpong 2>/dev/null | head -1)
    if [ -z "$RUN_DIR" ]; then
        echo "ERROR: 无法自动定位 BMN run_dir; 请显式传入第二个参数"
        exit 1
    fi
fi

# 校验 PID + run_dir
if ! ps -p "$TRAIN_PID" > /dev/null 2>&1; then
    echo "WARN: PID $TRAIN_PID 当前已不存在; 可能训练已结束, 直接进入 eval"
fi
if [ ! -d "$RUN_DIR" ]; then
    echo "ERROR: run_dir 不存在: $RUN_DIR"
    exit 1
fi
if [ ! -f "$RUN_DIR/manifest.json" ]; then
    echo "ERROR: manifest 缺失: $RUN_DIR/manifest.json"
    exit 1
fi

echo "[wait_and_eval] $(date -Iseconds)"
echo "  train_pid: $TRAIN_PID"
echo "  run_dir:   $RUN_DIR"
echo ""

# ---- 1. 等训练结束 ----
echo "[1/5] 等训练 PID $TRAIN_PID 结束..."
WAIT_START=$(date +%s)
while ps -p "$TRAIN_PID" > /dev/null 2>&1; do
    # 每 5 分钟打一次进度
    sleep 300
    NOW=$(date +%s)
    ELAPSED=$((NOW - WAIT_START))
    LATEST=$(tail -1 /tmp/bmn_full_train.log 2>/dev/null | grep -oE "epoch:\[[ 0-9/]+\] train step:[0-9]+ loss: [0-9.]+" || echo "(log unavailable)")
    echo "  [wait +${ELAPSED}s] $LATEST"
done
echo "[1/5] 训练已结束 (等待 $(($(date +%s) - WAIT_START))s)"
echo ""

# ---- 2. Finalize manifest (如果 cli/train.py 没自动 finalize) ----
echo "[2/5] Finalize manifest..."
.venv/bin/python <<PYEOF
import json
from pathlib import Path
from pingpong_av.experiment.run_manifest import finalize

run_dir = Path("$RUN_DIR")
m = json.loads((run_dir / "manifest.json").read_text())
if m.get("status") == "running":
    finalize(run_dir, status="completed")
    print(f"  → 已 finalize manifest")
else:
    print(f"  → manifest.status={m.get('status')!r}, 无需 finalize")
PYEOF
echo ""

# ---- 3. 找最大 epoch 的 ckpt ----
echo "[3/5] 找最终 ckpt..."
CKPT=$(ls -1 "$RUN_DIR"/BMN_epoch_*.pdparams 2>/dev/null | sort | tail -1)
if [ -z "$CKPT" ]; then
    echo "ERROR: $RUN_DIR 下没有 BMN_epoch_*.pdparams"
    exit 3
fi
echo "  ckpt: $CKPT"
echo ""

# ---- 4. 删 cached predictions, 备份旧 metrics ----
echo "[4/5] 清理 cached predictions, 备份旧 metrics..."
CACHED="$RUN_DIR/bmn_eval/results/bmn_results_validation.json"
if [ -f "$CACHED" ]; then
    rm -f "$CACHED"
    echo "  → 删 $CACHED"
fi
if [ -f "$RUN_DIR/metrics.json" ]; then
    cp "$RUN_DIR/metrics.json" "$RUN_DIR/metrics_pre_final.json"
    echo "  → 备份旧 metrics 到 metrics_pre_final.json"
fi
echo ""

# ---- 5. 跑最终 eval ----
echo "[5/5] 跑最终 eval (强制重新前向)..."
.venv/bin/pp eval --checkpoint "$CKPT" --split val --batch-size 4 \
    --output "$RUN_DIR/metrics_final.json"
EVAL_RC=$?
echo ""

if [ "$EVAL_RC" -ne 0 ]; then
    echo "ERROR: eval 失败 (rc=$EVAL_RC)"
    exit 3
fi

# 同步覆盖 metrics.json (让 manifest 里的 metrics 指向 final)
cp "$RUN_DIR/metrics_final.json" "$RUN_DIR/metrics.json"

echo "============================================================"
echo "✓ FINAL EVAL DONE  ($(date -Iseconds))"
echo "============================================================"
.venv/bin/python <<PYEOF
import json
from pathlib import Path
p = Path("$RUN_DIR/metrics_final.json")
m = json.loads(p.read_text())
ar = m.get("metrics", {})
print(f"  ckpt:     {Path(m['checkpoint']).name}")
print(f"  AR@1:     {ar.get('ar@1', 0):.2f}")
print(f"  AR@5:     {ar.get('ar@5', 0):.2f}")
print(f"  AR@10:    {ar.get('ar@10', 0):.2f}")
print(f"  AR@100:   {ar.get('ar@100', 0):.2f}")
print(f"  n_videos: {m.get('n_videos_evaluated')}")
print(f"  n_props:  {m.get('n_proposals')}")
print(f"  metrics_final.json: {p}")
PYEOF
echo "============================================================"
exit 0
