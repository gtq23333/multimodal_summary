from __future__ import annotations

import re

from rules.common.watermark import is_watermark_line

_GRADUATE_NOISE_RES: list[re.Pattern[str]] = [
    re.compile(r"^参赛队号"),
    re.compile(r"^参赛密码"),
    re.compile(r"^队员姓名"),
    re.compile(r"^队员\s*\d"),
    re.compile(r"^参赛队员"),
    re.compile(r"^学校\s"),
    re.compile(r"^#?\s*中国研究生创新实践系列大赛"),
    re.compile(r"^#?\s*.*华为杯.*研究生数学建模竞赛"),
    re.compile(r"^#?\s*全国第.+届研究生数学建模竞赛"),
    re.compile(r"^#?\s*题目\s*$"),
    re.compile(r"^<table>.*参赛队号", re.I),
    re.compile(r"^!\[\]\(images/"),
    re.compile(r"^（由组委会填写）"),
]

_OTHER_NOISE_RES: list[re.Pattern[str]] = [
    re.compile(r"^#?\s*参赛队号"),
    re.compile(r"^#?\s*报名号"),
    re.compile(r"^#?\s*参赛报名号"),
    re.compile(r"^#?\s*承诺书"),
    re.compile(r"^#?\s*编号专用页"),
    re.compile(r"^#?\s*竞赛统一编号"),
    re.compile(r"^#?\s*竞赛评阅编号"),
    re.compile(r"^#?\s*评阅记录"),
    re.compile(r"^队员\s*\d"),
    re.compile(r"^参赛队员"),
    re.compile(r"^参赛队教练员"),
    re.compile(r"^参赛队伍组别"),
    re.compile(r"^参赛密码"),
    re.compile(r"^所选题目"),
    re.compile(r"^#?\s*第四届.*互动出版杯"),
    re.compile(r"^#?\s*数学建模网络挑战赛"),
    re.compile(r"^#?\s*数学中国杯"),
    re.compile(r"^#?\s*五一数学建模联赛"),
    re.compile(r"^#?\s*深圳杯"),
    re.compile(r"^我们仔细阅读了"),
    re.compile(r"^我们完全明白"),
    re.compile(r"^我们知道，抄袭"),
    re.compile(r"^我们郑重承诺"),
    re.compile(r"^我们允许数学中国"),
    re.compile(r"^我们的参赛"),
    re.compile(r"^获奖证书邮寄地址"),
    re.compile(r"^日期："),
    re.compile(r"^所属学校"),
    re.compile(r"^指导老师"),
    re.compile(r"^队员："),
    re.compile(r"^指导教师"),
    re.compile(r"^!\[\]\(images/"),
    re.compile(r"^<table>"),
]

_NAME_LINE_RE = re.compile(
    r"^(?:队员\d|队员\s*\d|参赛队员|队员姓名|队员：).{0,30}$"
)


def _matches_noise(line: str, patterns: list[re.Pattern[str]]) -> bool:
    s = line.strip()
    if not s:
        return False
    if is_watermark_line(s):
        return True
    if _NAME_LINE_RE.match(s):
        return True
    for pat in patterns:
        if pat.search(s):
            return True
    return False


def strip_noise_lines(lines: list[str], *, profile: str) -> list[str]:
    patterns = _OTHER_NOISE_RES if profile == "other_competition" else _GRADUATE_NOISE_RES
    return [ln for ln in lines if not _matches_noise(ln, patterns)]
