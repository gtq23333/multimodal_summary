from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from profiles.national_competition import NationalCompetitionProfile  # noqa: E402
from rules.national_competition.compose import compose  # noqa: E402
from rules.national_competition.dedupe_stub import dedupe_11_stubs  # noqa: E402
from rules.national_competition.patterns import DEFAULT_SEPARATOR  # noqa: E402
from rules.national_competition.toc import strip_toc_block  # noqa: E402

FILLER = "本段为测试用补充正文，用于满足最小正文长度门槛，描述数据预处理、模型建立、求解算法与结果分析等完整流程，并讨论假设条件、参数设置与灵敏度分析。" * 2

SEP = DEFAULT_SEPARATOR
PROFILE = NationalCompetitionProfile()


STANDARD_RAW = f"""# 示例论文标题

# 摘要

这是摘要第一段，描述问题背景与总体方法，字数足够长以满足质量门槛要求，包含针对问题一、问题二的基本结论概述与模型方法说明。

针对问题一，我们建立了数学模型并求解得到主要结果，给出了完整的参数估计与灵敏度分析。

针对问题二，我们进一步扩展模型并给出优化方案，比较了多种算法在精度与效率上的差异。

# 关键词：示例 关键词

# 一、问题重述

# 1.1背景

1.1 背景资料这是正文的背景段落，内容足够长，用于说明问题场景与数据条件，满足正文长度检测需求，并描述系泊系统或类似工程场景中的关键参数与约束。

# 二、问题分析

二、问题分析本节分析各小问思路，讨论模型假设、符号说明以及求解方法的选择依据与改进方向，并给出各小问之间的逻辑关系与总体技术路线说明，补充灵敏度分析与模型评价内容。{FILLER}
"""

TOC_RAW = f"""# 标题

# 摘要

摘要正文足够长以满足质量门槛要求，包含针对问题一的简要说明与主要结论，确保字符数超过八十字符阈值，并补充方法概述。

针对问题一，建立模型并求解；针对问题二，扩展优化模型并给出数值结果与对比分析。

# 关键词：测试

# 目录

一、问题重述 3

二、问题分析 4

# 一、问题背景与重述

正文起始段落足够长，描述智能加工系统与 RGV 调度背景，满足正文最小长度要求，并说明题目数据与约束条件，包括加工参数、故障率与工序约束等细节。{FILLER}

# 二、模型假设

假设一：系统稳定运行。假设二：加工时间已知。假设三：故障过程满足马尔可夫性质。假设四：物料到达过程符合给定分布。{FILLER}
"""

STUB_RAW = f"""# 标题

# 摘要

摘要足够长，包含针对问题一与问题二的结论，字符数超过八十字符以满足清洗质量门槛检测，并概述主要建模思路。

针对问题一，完成数据预处理；针对问题二，建立优化模型并给出调度方案。

# 关键词：a b

# 一、问题重述

# 1.1背景

短 stub。

1.1 背景资料完整背景段落，内容远长于 stub，应保留此段作为正文起点后的首段，并继续描述问题背景与数据特征。{FILLER}

# 1.2需要解决的问题

问题列表一、问题列表二、问题列表三，分别对应不同小问的具体要求与约束条件说明。{FILLER}
"""


def test_compose_separator():
    out = compose("摘要段", "正文段", separator=SEP)
    assert SEP in out
    assert out.index("摘要段") < out.index(SEP) < out.index("正文段")


def test_strip_toc_block():
    lines = TOC_RAW.splitlines()
    idx = next(i for i, l in enumerate(lines) if l.strip() == "# 目录")
    rest, _ = strip_toc_block(lines, idx)
    joined = "\n".join(rest)
    assert "目录" not in joined.split("\n")[0]
    assert "正文起始段落" in joined


def test_dedupe_stub():
    lines = [
        "1.1 背景短 stub。",
        "1.1 背景资料完整背景段落，内容远长于 stub，应保留此段。",
        "其他内容。",
    ]
    out = dedupe_11_stubs(lines)
    assert len(out) == 2
    assert "完整背景" in out[0]


def test_standard_clean():
    result = PROFILE.clean(STANDARD_RAW, paper_id="test", separator=SEP)
    assert result.success, result.reason
    assert SEP in result.content
    assert "针对问题一" in result.abstract
    assert "1.1 背景资料" in result.body
    assert "# 摘要" not in result.content
    assert "# 目录" not in result.body


def test_toc_clean():
    result = PROFILE.clean(TOC_RAW, paper_id="test_toc", separator=SEP)
    assert result.success, result.reason
    assert "目录" not in result.body.split("\n")[0]
    assert "正文起始段落" in result.body


H2_ABSTRACT_RAW = f"""# 示例论文标题

## 摘要

这是摘要第一段，描述问题背景与总体方法，字数足够长以满足质量门槛要求，包含针对问题一、问题二的基本结论概述与模型方法说明。

针对问题一，我们建立了数学模型并求解得到主要结果，给出了完整的参数估计与灵敏度分析。

针对问题二，我们进一步扩展模型并给出优化方案，比较了多种算法在精度与效率上的差异。

关键词：示例 关键词

## 一、问题重述

## 1.1 问题背景

1.1 问题背景这是正文的背景段落，内容足够长，用于说明问题场景与数据条件，满足正文长度检测需求，并描述系泊系统或类似工程场景中的关键参数与约束。{FILLER}

## 二、问题分析

二、问题分析本节分析各小问思路，讨论模型假设、符号说明以及求解方法的选择依据与改进方向，并给出各小问之间的逻辑关系与总体技术路线说明，补充灵敏度分析与模型评价内容。{FILLER}
"""

SECTION_ONE_ABSTRACT_RAW = f"""# 圈舍养殖空间布局优化问题

# 一、摘要

本文研究某养殖品种，通过建立标准圈舍布局模型，结合人员配置与分阶段分群管理，在不同阶段、性别和大小分组只对空间要求不同的约束下，对每一阶段可养殖的母畜数量与公畜数量进行决策优化。

针对问题1，母畜的管理周期包括配种期、怀孕期、哺乳期和休整期，一个周期约为229天。我们建立以单个周期内母畜以固定周期的方式重复利用的整数规划模型(非线性规划)，占用面积最小化目标函数。

针对问题2，考虑多周期情形，建立具备周期之间间隔的模型，以母畜数量为目标函数，112个约束条件建立线性规划模型。

针对问题3，研究特定母畜和公畜配种时间的影响，配种时间各阶段的分布使得周期主要落在配种期、怀孕期、哺乳期对母畜进行分组管理，不同分支的决策各有不同，最终确定225天的代养期。

关键词：圈舍布局，标准圈舍，线性规划，MATLAB

# 二、问题重述

问题重述正文起始段落足够长，描述养殖场景与数据条件，满足正文最小长度要求，并说明题目数据与约束条件，包括加工参数、故障率与工序约束等细节。{FILLER}

# 三、模型假设

假设一：系统稳定运行。假设二：加工时间已知。{FILLER}
"""


def test_h2_abstract_clean():
    result = PROFILE.clean(H2_ABSTRACT_RAW, paper_id="test_h2", separator=SEP)
    assert result.success, result.reason
    assert "针对问题一" in result.abstract
    assert "1.1 问题背景" in result.body
    assert "## 摘要" not in result.content


def test_section_one_abstract_clean():
    result = PROFILE.clean(
        SECTION_ONE_ABSTRACT_RAW, paper_id="test_sec1_abs", separator=SEP
    )
    assert result.success, result.reason
    assert "针对问题1" in result.abstract
    assert "问题重述正文" in result.body
    assert "# 一、摘要" not in result.content


def test_stub_clean():
    result = PROFILE.clean(STUB_RAW, paper_id="test_stub", separator=SEP)
    assert result.success, result.reason
    assert "短 stub" not in result.body
    assert "完整背景段落" in result.body

