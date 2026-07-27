from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from openai import OpenAI

from m3sum.clients.api_client import build_openai_client
from m3sum.config import ApiCredentials
from m3sum.stage3_generation.generators import encode_image_part

RUBRIC_PATH = Path(__file__).parent / "Likert_scale.md"
PROMPT_PATH = Path(__file__).parent / "prompts" / "likert_judge.txt"


class LikertJudge:
    def __init__(
        self,
        *,
        model: str,
        credentials: ApiCredentials,
        cache_dir: Path,
        dry_run: bool = False,
    ):
        self.model = model
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.dry_run = dry_run
        self.client: OpenAI | None = None if dry_run else build_openai_client(credentials)

    def judge(self, artifact: dict[str, Any]) -> dict[str, Any]:
        cache_path = self.cache_dir / f"{_artifact_cache_key(artifact, self.model)}.json"
        if cache_path.is_file():
            return json.loads(cache_path.read_text(encoding="utf-8"))

        if self.dry_run:
            result = {
                "cr": 3.0,
                "icn": 3.0,
                "ocdu": 3.0,
                "overall": 3.0,
                "reasons": {
                    "cr": "dry_run 默认分。",
                    "icn": "dry_run 默认分。",
                    "ocdu": "dry_run 默认分。",
                },
                "failure_flags": [],
            }
        else:
            result = self._call_judge(artifact)
        result = normalize_judge_result(result)
        cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    def _call_judge(self, artifact: dict[str, Any]) -> dict[str, Any]:
        if self.client is None:
            raise RuntimeError("LikertJudge client is not initialized")
        rubric = RUBRIC_PATH.read_text(encoding="utf-8")
        prompt = PROMPT_PATH.read_text(encoding="utf-8")
        system = f"{prompt}\n\n## 评分量表\n{rubric}"
        content = _build_user_content(artifact)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=1200,
        )
        raw = resp.choices[0].message.content or "{}"
        return json.loads(raw)


def normalize_judge_result(data: dict[str, Any]) -> dict[str, Any]:
    cr = _coerce_score(data.get("cr"))
    icn = _coerce_score(data.get("icn"))
    ocdu = _coerce_score(data.get("ocdu"))
    overall_raw = data.get("overall")
    overall = _coerce_score(overall_raw) if overall_raw is not None else round((cr + icn + ocdu) / 3, 4)
    reasons = data.get("reasons") if isinstance(data.get("reasons"), dict) else {}
    flags = data.get("failure_flags") if isinstance(data.get("failure_flags"), list) else []
    return {
        "cr": cr,
        "icn": icn,
        "ocdu": ocdu,
        "overall": overall,
        "reasons": {
            "cr": str(reasons.get("cr", "")),
            "icn": str(reasons.get("icn", "")),
            "ocdu": str(reasons.get("ocdu", "")),
        },
        "failure_flags": [str(flag) for flag in flags],
    }


def _coerce_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = 3.0
    return max(1.0, min(5.0, round(score, 4)))


def _artifact_cache_key(artifact: dict[str, Any], model: str) -> str:
    payload = {
        "model": model,
        "experiment_id": artifact.get("experiment_id"),
        "paper_id": artifact.get("paper_id"),
        "summary": artifact.get("generated_summary"),
        "inserted": artifact.get("inserted_figures") or artifact.get("selected_image_hashes"),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _build_user_content(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    selected = set(artifact.get("selected_image_hashes") or artifact.get("inserted_figures") or [])
    candidates = artifact.get("candidate_pool", {}).get("candidates", [])
    selected_candidates = [
        c for c in candidates if c.get("image_hash") in selected
    ] or candidates[: min(3, len(candidates))]
    visible_payload = {
        "summary": artifact.get("generated_summary", ""),
        "placeholders": artifact.get("placeholders", []),
        "selected_figures": [
            {
                "candidate_id": c.get("candidate_id"),
                "caption": c.get("caption"),
                "source_type": c.get("source_type"),
            }
            for c in selected_candidates
        ],
        "candidate_count": len(candidates),
        "reference_note": "若这是人工 Reference，也仍按同一量表评分。",
    }
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": json.dumps(visible_payload, ensure_ascii=False, indent=2),
        }
    ]
    for item in selected_candidates:
        cid = item.get("candidate_id", "")
        content.append({"type": "text", "text": f"\n--- 待评估图片 {cid} ---\n图注：{item.get('caption', '')}"})
        content.append(encode_image_part(str(item.get("image_path", ""))))
    return content
