# Specification Quality Checklist: 原始视频到 BMN 时序定位的端到端推理与训练适配

**Purpose**: 在进入规划阶段之前验证规范的完整性和质量
**Created**: 2026-05-13
**Feature**: [Link to spec.md](../spec.md)

## Content Quality

- [x] No implementation details (languages, frameworks, APIs)
  - **Note**: 规范中提及 ffmpeg / PP-TSM / BMN / pkl / ndarray, 这些是**业务实体名**与**已存在的本仓库工件**, 不是新引入的实现细节. 类似 v0.1.x spec 中已经在用 "PaddleVideo / submodule / patches", 已成既有上下文.
- [x] Focused on user value and business needs
  - 三个 US 都是用户视角: 视频 → 时间区间 (US1), 我自己的数据扩库 (US2), 微调基线 (US3).
- [x] Written for non-technical stakeholders
  - US1/US2/US3 的"优先级原因 + 独立测试"都是业务语言.
- [x] All mandatory sections completed
  - 用户场景 / 边界情况 / 需求 / 关键实体 / 成功标准 / 假设 / 依赖, 全部填写.

## Requirement Completeness

- [x] No [NEEDS CLARIFICATION] markers remain
  - 全部用合理默认值填补, Session 2026-05-13 已记录 3 个事实型澄清.
- [x] Requirements are testable and unambiguous
  - FR-033 ~ FR-049 每条都可直接转化为单元/集成测试 (例如 FR-037: 单测验证 .shape[1] == 2048).
- [x] Success criteria are measurable
  - SC-010 (≤ 5 min 端到端) / SC-011 (≥ 80 帧/秒) / SC-012 (cosine ≥ 0.99) / SC-013 (cosine ≥ 0.999) / SC-014 (AR@100 不低于 -2%) — 全部含具体阈值.
- [x] Success criteria are technology-agnostic (no implementation details)
  - SC-010 描述用户体验 ("用 1 条命令" / "≤ 5 min"), 不限定具体框架.
  - SC-011 / SC-012 / SC-013 是数值结果, 不限定怎么实现.
- [x] All acceptance scenarios are defined
  - US1 三个场景 (正常 / 缺权重 / 视频损坏) / US2 三个场景 (正常 / 性能 / 幂等) / US3 一个场景 (resume 起作用).
- [x] Edge cases are identified
  - fps 不匹配 / 视频过短 / OOM / GT JSON 缺视频 / 权重 sha256 不匹配 / 帧数偏差.
- [x] Scope is clearly bounded
  - "不在范围内" 段明确划界: 不新加 patches / 不流式 / 不音频多模态 / 不换标签体系 / 不换特征提取器.
- [x] Dependencies and assumptions identified
  - "依赖" 段说清前置 (v0.2.x), "假设" 段说清环境/URL/权重等价性.

## Feature Readiness

- [x] All functional requirements have clear acceptance criteria
  - 每个 FR 都有对应的 US 验收场景或 SC 阈值.
- [x] User scenarios cover primary flows
  - P1 闭环推理 + P2 数据扩充 + P3 微调, 三个优先级覆盖业务全主线.
- [x] Feature meets measurable outcomes defined in Success Criteria
  - SC-010 直接对应 US1; SC-011 对应 US2; SC-014 对应 US3; 全部有可执行测试路径.
- [x] No implementation details leak into specification
  - 规范中描述的 schema (例如 `image_feature` ndarray (N, 2048)) 是**已经在数据集里存在的事实**, 不是本 feature 的设计决策. 列出来是为了避免歧义.

## Notes

- ✅ 全部清单项通过, 无需迭代
- 规范基于实地探测 (`Agent` Explore agent 在 PaddleVideo submodule 中确认了所有 BCEBOS URL / 配置路径 / output index / fps), 不是凭空推测
- 关键技术决策 (取 `output_names[1]` 而不是 `[0]`) 被显式提到 FR-037, 防止实现阶段误用
- 与现有 v0.2.x 代码的关系明确: FR-042 强制复用现有 BMN 路径, 不允许重复实现; FR-045 明确训练侧零改动
- 准备好进入 `/speckit.clarify` (可选, 但本规范已自洽) 或 `/speckit.plan`
