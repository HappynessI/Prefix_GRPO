from __future__ import annotations

from typing import Any, Dict, Tuple


def resolve_eval_row(row: Dict[str, Any], env_name: str) -> Tuple[str, int, Dict[str, Any]]:
    extra_info = row.get("extra_info") or {}
    interaction_kwargs = dict(extra_info.get("interaction_kwargs") or {})
    if not interaction_kwargs:
        raise ValueError(f"{env_name} eval row is missing extra_info.interaction_kwargs")

    interaction_kwargs.setdefault("name", env_name)

    session_id = interaction_kwargs.get("session_id")
    if session_id is None:
        raise ValueError(f"{env_name} eval row is missing interaction_kwargs.session_id")
    session_id = int(session_id)
    interaction_kwargs["session_id"] = session_id

    item_id = row.get("item_id") or interaction_kwargs.get("official_item_id")
    if item_id is None:
        item_id = f"{env_name}_{session_id}"

    return str(item_id), session_id, interaction_kwargs
