"""End-to-end annotation export validation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

base_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(base_dir))

from export.annotation_exporter import export_annotation
from export.annotation_loader import restore_from_file, used_image_hashes
from loaders.paper_loader import PaperLoader
from models.segments import TextSegment, insert_image_at_index, split_text_segment


def annotate_and_export(paper_id: str, loader: PaperLoader, output_dir: Path) -> Path:
    paper, err = loader.load_paper(paper_id)
    assert err is None and paper is not None

    segments = [TextSegment(content=paper.abstract_text)]
    if len(paper.filtered_candidates) >= 2:
        c1, c2 = paper.filtered_candidates[0], paper.filtered_candidates[1]
        segments, at1 = split_text_segment(segments, 0, 100)
        segments = insert_image_at_index(segments, at1, c1)
        segments, at2 = split_text_segment(segments, 0, 300)
        segments = insert_image_at_index(segments, at2, c2)
    elif paper.filtered_candidates:
        c = paper.filtered_candidates[0]
        segments, at = split_text_segment(segments, 0, len(paper.abstract_text) // 2)
        segments = insert_image_at_index(segments, at, c)

    out_path = export_annotation(
        output_dir=output_dir,
        paper=paper,
        segments=segments,
        original_abstract=paper.abstract_text,
        filter_strategy="body_with_caption",
        context_window_chars=30,
        tool_version="0.1.0",
    )

    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["schema_version"] == "1.0.0"
    assert data["paper_id"] == paper_id
    assert "insertions" in data
    assert "multimodal_sequence" in data

    restored = restore_from_file(paper, out_path)
    assert used_image_hashes(restored) == used_image_hashes(segments)

    print(f"[OK] E2E {paper_id}: insertions={len(data['insertions'])}, saved to {out_path.name}")
    return out_path


def main():
    config_path = base_dir / "config.yaml"
    loader = PaperLoader(config_path)
    output_dir = base_dir / "annotations" / "_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    annotate_and_export("2016_G_A433.pdf-ace5f580-0291-4b74-8e8a-3e24514c4563", loader, output_dir)
    annotate_and_export("2017_G_D018.pdf-7c0f51ce-2a33-442e-94f9-7b16048bc8f7", loader, output_dir)

    print("\nE2E tests passed.")


if __name__ == "__main__":
    main()
