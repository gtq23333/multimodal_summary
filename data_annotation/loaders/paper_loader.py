from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from filters.registry import get_filter_strategy
from loaders.content_list_loader import find_content_list_path, load_candidates_from_content_list
from loaders.md_loader import parse_md
from models.paper import Paper


@dataclass
class LoadError:
    paper_id: str
    reason: str


class PaperLoader:
    def __init__(self, config_path: Path):
        with open(config_path, encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        base = config_path.parent
        paths = self.config["paths"]
        self.md_corpus_root = (base / paths["md_corpus_root"]).resolve()
        self.mineru_corpus_root = Path(paths["mineru_corpus_root"]).resolve()
        self.abstract_separator = self.config["annotation"]["abstract_separator"]
        self.filter_strategy = get_filter_strategy(self.config.get("image_filter", {}))

    def list_paper_ids(self) -> list[str]:
        md_files = sorted(self.md_corpus_root.glob("*.md"))
        return [f.stem for f in md_files]

    def load_paper(self, paper_id: str) -> tuple[Paper | None, LoadError | None]:
        md_path = self.md_corpus_root / f"{paper_id}.md"
        if not md_path.is_file():
            return None, LoadError(paper_id=paper_id, reason="md_not_found")

        mineru_dir = self.mineru_corpus_root / paper_id
        if not mineru_dir.is_dir():
            return None, LoadError(paper_id=paper_id, reason="mineru_dir_not_found")

        content_list_path = find_content_list_path(mineru_dir)
        if content_list_path is None:
            return None, LoadError(paper_id=paper_id, reason="content_list_not_found")

        abstract_text, body_text, body_hashes = parse_md(md_path, self.abstract_separator)
        all_candidates = load_candidates_from_content_list(content_list_path, mineru_dir, body_hashes)

        paper = Paper(
            paper_id=paper_id,
            md_path=md_path,
            mineru_dir=mineru_dir,
            content_list_path=content_list_path,
            abstract_text=abstract_text,
            body_text=body_text,
            body_image_hashes=body_hashes,
            all_candidates=all_candidates,
        )
        paper.filtered_candidates = self.filter_strategy.filter(all_candidates, paper)
        return paper, None
