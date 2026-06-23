from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Union

from models.paper import ImageCandidate


@dataclass
class TextSegment:
    type: Literal["text"] = "text"
    content: str = ""


@dataclass
class ImageSegment:
    candidate: ImageCandidate
    type: Literal["image"] = "image"


Segment = Union[TextSegment, ImageSegment]


def plain_text_from_segments(segments: list[Segment]) -> str:
    return "".join(s.content for s in segments if s.type == "text")


def compute_char_index(segments: list[Segment], segment_index: int, offset_in_text: int = 0) -> int:
    char_index = 0
    for i, seg in enumerate(segments):
        if i >= segment_index:
            break
        if seg.type == "text":
            char_index += len(seg.content)
    if segment_index < len(segments) and segments[segment_index].type == "text":
        char_index += offset_in_text
    return char_index


def compute_context(plain_text: str, char_index: int, window: int) -> tuple[str, str]:
    before = plain_text[max(0, char_index - window):char_index]
    after = plain_text[char_index:char_index + window]
    return before, after


def split_text_segment(segments: list[Segment], seg_index: int, offset: int) -> tuple[list[Segment], int]:
    """Split text segment at offset; return new segments list and index of inserted image slot."""
    seg = segments[seg_index]
    if seg.type != "text":
        raise ValueError("Can only split text segments")
    before = seg.content[:offset]
    after = seg.content[offset:]
    new_segments: list[Segment] = segments[:seg_index]
    if before:
        new_segments.append(TextSegment(content=before))
    insert_index = len(new_segments)
    if after:
        new_segments.append(TextSegment(content=after))
    new_segments.extend(segments[seg_index + 1:])
    return new_segments, insert_index


def insert_image_at_index(segments: list[Segment], insert_index: int, candidate: ImageCandidate) -> list[Segment]:
    new_segments = segments[:insert_index] + [ImageSegment(candidate=candidate)] + segments[insert_index:]
    return new_segments


def remove_image_segment(segments: list[Segment], seg_index: int) -> tuple[list[Segment], ImageCandidate]:
    seg = segments[seg_index]
    if seg.type != "image":
        raise ValueError("Not an image segment")
    new_segments = segments[:seg_index] + segments[seg_index + 1:]
    return new_segments, seg.candidate


def merge_adjacent_text_segments(segments: list[Segment]) -> list[Segment]:
    merged: list[Segment] = []
    for seg in segments:
        if seg.type == "text" and merged and merged[-1].type == "text":
            merged[-1].content += seg.content
        else:
            merged.append(TextSegment(content=seg.content) if seg.type == "text" else ImageSegment(candidate=seg.candidate))
    return merged
