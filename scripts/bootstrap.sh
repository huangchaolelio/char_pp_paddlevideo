#!/usr/bin/env bash
# ============================================================
# bootstrap.sh — 项目隔离环境一键初始化 (章程 VII / VIII)
# ============================================================
# 目标: 让一名首次 clone 仓库的工程师执行此脚本即可获得:
#   1) 项目独立的 .venv (Python 3.11)
#   2) 业务代码依赖 (requirements/base.txt) 安装到位
#   3) 上游 PaddleVideo + paddlepaddle-gpu 安装与 3.11 适配补丁应用
#
# 章程对齐:
#   - 章程 VII: 端到端 ≤ 5 条命令; 本脚本是 quickstart 的 "命令 0" (一次性初始化)
#   - 章程 VIII: 严禁使用系统 Python; .venv 强制基于 python3.11
#   - 章程 VI:  上游通过 submodule + patches 接入, 不复制源码
#
# 退出码:
#   0  成功
#   2  环境问题 (python3.11 缺失 / submodule 未初始化等)
#   4  运行时失败 (pip install 失败等)
#
# 用法:
#   bash scripts/bootstrap.sh              # 标准初始化
#   bash scripts/bootstrap.sh --smoke      # 完成后追加 env-check --strict 自检
#   bash scripts/bootstrap.sh --skip-upstream  # 仅装业务依赖, 不装 paddle/paddlevideo (调试用)
# ============================================================

set -euo pipefail

# ---- 颜色 (仅在终端) ----
if [[ -t 1 ]]; then
    GREEN='\033[1;32m'; YELLOW='\033[1;33m'; RED='\033[1;31m'; CYAN='\033[1;36m'; NC='\033[0m'
else
    GREEN=''; YELLOW=''; RED=''; CYAN=''; NC=''
fi

log()  { printf '%b[bootstrap]%b %s\n' "$CYAN"   "$NC" "$*"; }
ok()   { printf '%b[bootstrap]%b %s\n' "$GREEN"  "$NC" "$*"; }
warn() { printf '%b[bootstrap]%b %s\n' "$YELLOW" "$NC" "$*" >&2; }
err()  { printf '%b[bootstrap]%b %s\n' "$RED"    "$NC" "$*" >&2; }

# ---- 参数 ----
DO_SMOKE=0
SKIP_UPSTREAM=0
for arg in "$@"; do
    case "$arg" in
        --smoke)         DO_SMOKE=1 ;;
        --skip-upstream) SKIP_UPSTREAM=1 ;;
        -h|--help)
            sed -n '1,/^# =\+/p; /^# 用法:/,/^# =\+/p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            err "未知参数: $arg"
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

VENV_DIR="$REPO_ROOT/.venv"
VENV_PY="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"

# ============================================================
# Step 1 — 探测 python3.11
# ============================================================
log "Step 1/5: 探测 python3.11 ..."
PY311_BIN=""
for cand in python3.11 python3.11.9 python3.11.8 python3.11.7; do
    if command -v "$cand" >/dev/null 2>&1; then
        ver="$("$cand" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
        if [[ "$ver" == "3.11" ]]; then
            PY311_BIN="$(command -v "$cand")"
            break
        fi
    fi
done

if [[ -z "$PY311_BIN" ]]; then
    err "未找到 python3.11 可执行程序. 章程 VIII 要求本项目锁定 Python 3.11."
    err "请先安装 Python 3.11 (apt/brew/pyenv 任一), 然后重新运行本脚本."
    exit 2
fi
ok  "找到 python3.11: $PY311_BIN ($("$PY311_BIN" --version 2>&1))"

# ============================================================
# Step 2 — 创建/复用 .venv
# ============================================================
log "Step 2/5: 准备 .venv ..."
if [[ -d "$VENV_DIR" ]]; then
    if [[ -x "$VENV_PY" ]]; then
        existing_ver="$("$VENV_PY" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "?")"
        if [[ "$existing_ver" == "3.11" ]]; then
            ok "复用已有 .venv ($existing_ver)"
        else
            warn ".venv 现有 Python 版本为 $existing_ver, 与 3.11 不符, 重建中..."
            rm -rf "$VENV_DIR"
            "$PY311_BIN" -m venv "$VENV_DIR"
            ok "已重建 .venv"
        fi
    else
        warn ".venv 目录损坏 (缺少 python 可执行), 重建中..."
        rm -rf "$VENV_DIR"
        "$PY311_BIN" -m venv "$VENV_DIR"
        ok "已重建 .venv"
    fi
else
    "$PY311_BIN" -m venv "$VENV_DIR"
    ok "已创建 .venv"
fi

# 升级 pip 自身, 避免老 pip 解析新 wheel 失败
"$VENV_PIP" install --upgrade --quiet pip setuptools wheel

# ============================================================
# Step 3 — 业务依赖 (requirements/base.txt)
# ============================================================
log "Step 3/5: 安装业务依赖 (requirements/base.txt) ..."
"$VENV_PIP" install --quiet -r "$REPO_ROOT/requirements/base.txt"
ok "业务依赖安装完成"

# ============================================================
# Step 4 — 上游依赖与 PaddleVideo (T010 完善)
# ============================================================
if [[ "$SKIP_UPSTREAM" -eq 1 ]]; then
    warn "Step 4/5: 跳过上游安装 (--skip-upstream)"
else
    log "Step 4/5: 安装上游 PaddlePaddle-GPU 与 PaddleVideo ..."

    # 4a) submodule 是否已 init
    if [[ ! -f "$REPO_ROOT/third_party/PaddleVideo/setup.py" ]]; then
        log "  · 初始化 submodule (third_party/PaddleVideo)..."
        git -C "$REPO_ROOT" submodule update --init --recursive
    fi

    # 4b) 上游适配依赖 (paddlepaddle-gpu + 3.11 兼容版本)
    "$VENV_PIP" install --quiet -r "$REPO_ROOT/requirements/upstream-py311.txt"
    ok "  · 上游 Python 依赖安装完成"

    # 4c) 应用 third_party/patches/*.patch 到 submodule 工作区
    log "  · 应用 3.11 兼容补丁 ..."
    bash "$REPO_ROOT/scripts/apply_upstream_patches.sh"

    # 4d) PaddleVideo 接入策略: **不**通过 pip install -e 上游仓库 (章程 VI / R3 调研结论).
    #
    # 原因 (T033 实测发现):
    #   1) 上游 setup.py 以 `requirements.txt` 形式声明 install_requires, 而该文件钉死了
    #      `opencv-python==4.2.0.32` / `decord==0.4.2` / `scipy==1.6.3` / `av==8.0.3` 等
    #      Python 3.11 下**没有 wheel 的版本**, 导致 pip install -e 必然失败.
    #   2) 上游 setup.py 把 `paddlevideo/` 子目录重映射成名为 `ppvideo` 的 package
    #      (package_dir={'ppvideo': ''}), 即便安装成功, `import paddlevideo` 也找不到模块.
    #
    # 替代方案 (与章程 VI "上游最小侵入" 完全一致):
    #   - 不修改上游 setup.py / requirements.txt;
    #   - 不通过 pip 装上游;
    #   - 通过 src/pingpong_av/upstream_adapter/importer.py 在运行时把
    #     `third_party/PaddleVideo` 加入 sys.path, 直接 import 上游的 `paddlevideo` 包.
    #   - 上游运行时所需的 Python 依赖已经在 4b 步骤通过本项目的
    #     requirements/upstream-py311.txt (3.11-compat overrides) 装齐.
    ok "  · 跳过 pip install -e PaddleVideo (使用 sys.path 兜底导入, 章程 VI / R3)"
fi

# ============================================================
# Step 5 — 把本仓库以 editable 装入, 让 `pp` 命令可用
# ============================================================
log "Step 5/5: 安装 pingpong_av 本身 (editable, 提供 \`pp\` 命令) ..."
"$VENV_PIP" install --quiet -e "$REPO_ROOT"
ok "本仓库安装完成; \`pp --help\` 现在可用 (需先激活 .venv)"

# ============================================================
# 完成提示
# ============================================================
echo
ok "✓ bootstrap 全部完成"
echo
echo -e "${CYAN}下一步:${NC}"
echo "  source .venv/bin/activate"
echo "  pp env-check --strict"
echo "  # 或者直接调用: .venv/bin/pp env-check --strict"
echo
echo -e "${CYAN}章程提醒 (章程 VIII):${NC}"
echo "  所有后续命令必须通过 .venv/bin/pp 或激活 .venv 后调用,"
echo "  禁止使用系统 Python."

# ============================================================
# 可选: --smoke 自检
# ============================================================
if [[ "$DO_SMOKE" -eq 1 ]]; then
    echo
    log "[--smoke] 运行 env-check --strict ..."
    if "$VENV_DIR/bin/pp" env-check --strict; then
        ok "✓ env-check 通过"
    else
        err "env-check 失败, 请按上面提示排查"
        exit 4
    fi
fi
