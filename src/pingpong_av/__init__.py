"""pingpong_av — 基于 PaddleVideo 的乒乓球视频动作识别业务包.

- 入口: CLI `pp` (见 pyproject.toml, `pingpong_av.cli:main`).
- 运行环境要求: Python 3.11 隔离 venv (章程 VIII).
- 业务代码与上游 PaddleVideo 严格分离 (章程 VI);
  对上游的单点接入放在 `pingpong_av.upstream_adapter`.
"""

__version__ = "0.1.0"
