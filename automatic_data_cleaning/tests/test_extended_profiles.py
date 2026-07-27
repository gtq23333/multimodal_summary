from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from profiles.graduate_competition import GraduateCompetitionProfile  # noqa: E402
from profiles.other_competition import OtherCompetitionProfile  # noqa: E402
from rules.national_competition.compose import compose  # noqa: E402
from rules.national_competition.patterns import DEFAULT_SEPARATOR  # noqa: E402

FILLER = (
    "本段为测试用补充正文，用于满足最小正文长度门槛，描述数据预处理、模型建立、"
    "求解算法与结果分析等完整流程，并讨论假设条件、参数设置与灵敏度分析。"
) * 2

SEP = DEFAULT_SEPARATOR
GRAD_PROFILE = GraduateCompetitionProfile()
OTHER_PROFILE = OtherCompetitionProfile()


GRADUATE_RAW = f"""# 中国研究生创新实践系列大赛

<table><tr><td>学校</td><td>武汉大学</td></tr></table>

# 题目 移动场景超分辨定位问题

# 摘 要：

移动场景的超分辨率定位具有极大的应用价值，然而现有的面向频连续波雷达 FMCW 的算法皆不同时具备低复杂度以及高分辨率这两个特征。

针对问题一，根据提供的无噪声仿真数据，建立了超分辨定位模型并求解主要结果，给出了完整的参数估计与灵敏度分析。

针对问题二，在问题一基础上针对环境噪声场景进行了参数调优和改进，最终选取窗口尺寸为(20，10)时效果最佳。

针对问题三，充分利用物体运动的连续性对搜索范围进行预测，实现了在线低复杂度的连续跟踪算法。

关键词：超分辨定位，2D-MUSIC 算法，FMCW-MUSIC 算法，移动场景

# 目录

1. 问题重述与分析 3

# 一、问题重述

# 1.1 问题背景

1.1 问题背景这是正文的背景段落，内容足够长，用于说明问题场景与数据条件，满足正文长度检测需求，并描述移动场景中的关键参数与约束。{FILLER}

# 二、模型假设

假设一：系统稳定运行。假设二：加工时间已知。{FILLER}

# 参考文献

[1] 示例文献
"""

OTHER_RAW = f"""# 参赛队号#1199

# 数学建模网络挑战赛

# 承诺书

我们仔细阅读了第四届数学建模网络挑战赛的竞赛规则。

队员1：徐冬
队员2：王鹏飞

# 2011年第四届数学建模网络挑战赛

题目 关于大型客机水上迫降问题的研究

关键词 客机迫降，最优姿态

# 摘 要：

该问题要求我们研究大型客机水上迫降的安全性问题。如果大型客机在空中飞行途中因发动机失灵只能在水上迫降，迫降时会给飞机产生很大的水体冲击力。

对于合理的保费浮动方案的建立，本文通过对所有数据的综合分析，根据影响保费的多种因素建立了数学模型。

参赛队号 1199

# 1.问题的重述

1. 问题的重述近几年来交通运输作为人们生活中的重要组成部分发挥着至关重要的作用，本文建立数学模型分析公交线路网络效率体系。{FILLER}

# 2.模型的假设

假设一：不考虑出行者从最初出发点到达第一个公交站点的步行。{FILLER}

# 参考文献

[1] 示例文献
"""

WATERMARK_RAW = f"""# 摘 要：

公众号：建模忠哥
获取更多资源
QQ群：966535540
本文研究某问题，通过建立数学模型对延迟退休年龄指标进行分析，用于解决当前延迟退休影响的问题，并提出合理化建议。

针对问题一，建立多元线性回归模型并给出主要结论，字符数超过八十字符以满足清洗质量门槛检测。

针对问题二，建立评价模型并通过层次分析法求出各群体的权重，给出延迟退休年限建议。

关键词：延迟退休，线性回归，层次分析

# 一、问题重述

问题重述正文起始段落足够长，描述养殖场景与数据条件，满足正文最小长度要求，并说明题目数据与约束条件。{FILLER}

# 二、模型假设

假设一：系统稳定运行。{FILLER}
"""


def test_compose_separator():
    out = compose("摘要段", "正文段", separator=SEP)
    assert SEP in out
    assert out.index("摘要段") < out.index(SEP) < out.index("正文段")


def test_graduate_clean():
    result = GRAD_PROFILE.clean(GRADUATE_RAW, paper_id="test_grad", separator=SEP)
    assert result.success, result.reason
    assert SEP in result.content
    assert "针对问题一" in result.abstract
    assert "1.1 问题背景" in result.body
    assert "建模忠哥" not in result.content
    assert "目录" not in result.body.split("\n")[0]


def test_other_clean():
    result = OTHER_PROFILE.clean(OTHER_RAW, paper_id="1199A.pdf-test", separator=SEP)
    assert result.success, result.reason
    assert SEP in result.content
    assert "水上迫降" in result.abstract
    assert "徐冬" not in result.content
    assert "承诺书" not in result.body
    assert "模型的假设" in result.body


def test_watermark_removed():
    result = GRAD_PROFILE.clean(WATERMARK_RAW, paper_id="wm_test", separator=SEP)
    assert result.success, result.reason
    assert "建模忠哥" not in result.content
    assert "966535540" not in result.content
