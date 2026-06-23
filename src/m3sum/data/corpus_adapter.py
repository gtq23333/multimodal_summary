from __future__ import annotations

import sys
from pathlib import Path

from m3sum.config import PipelineConfig
from m3sum.data.block_segmenter import enrich_figures, segment_body
from m3sum.data.problem_resolver import load_problem_text
from m3sum.data.schema import DocumentBundle


def _ensure_data_annotation_path(annotation_config: Path) -> None:
    da_root = annotation_config.parent
    if str(da_root) not in sys.path:
        sys.path.insert(0, str(da_root))


class CorpusAdapter:
    def __init__(self, config: PipelineConfig):
        self.config = config
        _ensure_data_annotation_path(config.annotation_config)
        from loaders.paper_loader import PaperLoader

        self._loader = PaperLoader(config.annotation_config)

    def load_paper(self, paper_id: str):
        paper, err = self._loader.load_paper(paper_id)
        if err is not None:
            raise RuntimeError(f"Failed to load {paper_id}: {err.reason}")
        return paper

    def load_document(self, paper_id: str) -> DocumentBundle:
        paper = self.load_paper(paper_id)
        blocks = segment_body(paper.body_text)
        figures = enrich_figures(paper.filtered_candidates, blocks)
        problem_text, _ = load_problem_text(self.config.problem_mds_root, paper_id)
        return DocumentBundle(
            paper_id=paper_id,
            abstract_text=paper.abstract_text,
            body_text=paper.body_text,
            blocks=blocks,
            figures=figures,
            problem_text=problem_text,
        )
