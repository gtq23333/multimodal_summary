"""Data layer validation for multimodal summary annotation tool."""

from __future__ import annotations

import sys
from pathlib import Path

base_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(base_dir))

from export.annotation_exporter import build_annotation
from loaders.paper_loader import PaperLoader
from models.segments import ImageSegment, TextSegment, insert_image_at_index, plain_text_from_segments, split_text_segment


def test_paper(paper_id: str, loader: PaperLoader) -> None:
    paper, err = loader.load_paper(paper_id)
    assert err is None, f"{paper_id}: {err}"
    assert paper is not None
    assert paper.abstract_text.strip(), f"{paper_id}: empty abstract"
    print(f"[OK] {paper_id}")
    print(f"     abstract chars: {len(paper.abstract_text)}")
    print(f"     body image hashes: {len(paper.body_image_hashes)}")
    print(f"     all candidates: {len(paper.all_candidates)}")
    print(f"     filtered candidates: {len(paper.filtered_candidates)}")
    for c in paper.filtered_candidates[:3]:
        print(f"       - [{c.source_type}] {c.caption[:40]} ({c.image_hash[:8]}...)")
    if len(paper.filtered_candidates) > 3:
        print(f"       ... and {len(paper.filtered_candidates) - 3} more")


def test_export_roundtrip(paper_id: str, loader: PaperLoader) -> None:
    paper, err = loader.load_paper(paper_id)
    assert err is None and paper is not None

    segments = [TextSegment(content=paper.abstract_text)]
    if paper.filtered_candidates:
        c = paper.filtered_candidates[0]
        mid = len(paper.abstract_text) // 2
        segments, insert_at = split_text_segment(segments, 0, mid)
        segments = insert_image_at_index(segments, insert_at, c)

    data = build_annotation(
        paper=paper,
        segments=segments,
        original_abstract=paper.abstract_text,
        filter_strategy="body_with_caption",
        context_window_chars=30,
        tool_version="0.1.0",
    )
    assert data["paper_id"] == paper_id
    assert len(data["insertions"]) == (1 if paper.filtered_candidates else 0)
    assert data["multimodal_sequence"]
    plain = plain_text_from_segments(segments)
    assert data["abstract"]["edited_text"] == plain
    if data["insertions"]:
        ins = data["insertions"][0]
        assert "char_index" in ins["position"]
        assert "text_before" in ins["position"]
        assert "text_after" in ins["position"]
    print(f"[OK] export roundtrip for {paper_id}, insertions={len(data['insertions'])}")


def main():
    config_path = base_dir / "config.yaml"
    loader = PaperLoader(config_path)

    test_paper("2016_G_A433.pdf-ace5f580-0291-4b74-8e8a-3e24514c4563", loader)
    test_paper("2017_G_D018.pdf-7c0f51ce-2a33-442e-94f9-7b16048bc8f7", loader)

    test_export_roundtrip("2016_G_A433.pdf-ace5f580-0291-4b74-8e8a-3e24514c4563", loader)
    test_export_roundtrip("2017_G_D018.pdf-7c0f51ce-2a33-442e-94f9-7b16048bc8f7", loader)

    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
