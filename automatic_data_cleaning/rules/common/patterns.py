from __future__ import annotations

import re

MD_HEAD = r"#{1,3}\s*"
CN_NUM = r"[一二三四五六七八九十百千]+"

ABSTRACT_TOKEN = r"摘\s*要"
KEYWORD_TOKEN = r"(?:关键词|关键字|\[关键词\])"

ABSTRACT_HEADING_RE = re.compile(
    rf"^{MD_HEAD}(?:{CN_NUM}[、．.\s]*)?(?:【)?{ABSTRACT_TOKEN}(?:】)?\s*[:：]?\s*$"
)
ABSTRACT_INLINE_RE = re.compile(rf"^{ABSTRACT_TOKEN}\s*[:：]?\s*")
KEYWORDS_RE = re.compile(rf"^(?:{MD_HEAD})?{KEYWORD_TOKEN}")
KEYWORDS_INLINE_RE = re.compile(rf"{KEYWORD_TOKEN}\s*[:：]")

TOC_HEADING_RE = re.compile(rf"^{MD_HEAD}目\s*录\s*$")
SECTION_CN_RE = re.compile(rf"^{MD_HEAD}{CN_NUM}[、．.\s]*\s*.+")
SECTION_NUM_RE = re.compile(rf"^{MD_HEAD}[0-9]+[、．.\s]*\s*.+")
SECTION_11_RE = re.compile(r"^(?:#{1,3}\s*)?1\.1\s*")
SECTION_11_HEADER_RE = re.compile(rf"^{MD_HEAD}1\.1\s*")

BODY_SECTION_ONE_RE = re.compile(
    rf"^{MD_HEAD}{CN_NUM}[、．.\s]*\s*(?:问题重述|问题背景(?:与重述)?|重述问题|问题的重述|问题)"
)
BODY_SECTION_NUM_RE = re.compile(
    rf"^{MD_HEAD}[0-9]+[、．.\s]\s*(?:问题重述|问题的重述|问题提出|问题背景|研究背景|背景知识)"
)
BODY_SECTION_PLAIN_RE = re.compile(
    rf"^{MD_HEAD}(?:问题重述|问题的重述|问题提出|问题背景)"
)

REFERENCES_HEADING_RE = re.compile(
    rf"^{MD_HEAD}(?:参考文献|参\s*考\s*文\s*献|References)\s*$",
    re.I,
)
ABSTRACT_END_SECTION_RE = re.compile(
    rf"^{MD_HEAD}(?:{CN_NUM}[、．.\s]*\s*)?(?:问题重述|问题的重述|问题分析|模型假设|符号说明|绪论)"
)
ENGLISH_ABSTRACT_RE = re.compile(
    rf"^{MD_HEAD}(?:英文摘要|Abstract)(?:\s*\(.*?\))?\s*$",
    re.I,
)

PAGE_TAIL_RE = re.compile(r"\d+\s*$")
PAGE_REF_RE = re.compile(r"\d+\.\d+")
DOT_LEADER_RE = re.compile(r"\.{4,}|…{2,}|·{4,}")

DEFAULT_SEPARATOR = "##############"
