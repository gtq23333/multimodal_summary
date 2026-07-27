from __future__ import annotations

import re

# minerU 导出常见 # / ## / ### 标题层级；国赛摘要偶见「# 一、摘要」
MD_HEAD = r"#{1,3}\s*"
CN_NUM = r"[一二三四五六七八九十百千]+"

ABSTRACT_HEADING_RE = re.compile(
    rf"^{MD_HEAD}(?:{CN_NUM}[、．.\s]*)?(?:【)?摘\s*要(?:】)?\s*[:：]?\s*$"
)
ABSTRACT_INLINE_RE = re.compile(r"^摘\s*要")
KEYWORDS_RE = re.compile(r"^(?:#{1,3}\s*)?(?:关键词|关键字|\[关键词\])")
TOC_HEADING_RE = re.compile(rf"^{MD_HEAD}目\s*录\s*$")
SECTION_CN_RE = re.compile(rf"^{MD_HEAD}{CN_NUM}[、．.\s]\s*.+")
SECTION_NUM_RE = re.compile(rf"^{MD_HEAD}[0-9]+[、．.\s]\s*.+")
SECTION_11_RE = re.compile(r"^(?:#{1,3}\s*)?1\.1\s*")
SECTION_11_HEADER_RE = re.compile(rf"^{MD_HEAD}1\.1\s*")
BODY_SECTION_ONE_RE = re.compile(
    rf"^{MD_HEAD}一[、．.\s]\s*(?:问题重述|问题背景(?:与重述)?|重述问题|问题)"
)
PAGE_TAIL_RE = re.compile(r"\d+\s*$")
PAGE_REF_RE = re.compile(r"\d+\.\d+")

DEFAULT_SEPARATOR = "##############"
