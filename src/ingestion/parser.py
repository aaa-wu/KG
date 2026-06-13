"""多格式数据解析器：支持 JSON/CSV/Excel"""
import json
import csv
import os
from typing import Any, Optional

VALID_ENTITY_LABELS = {"Major", "Course", "KnowledgeConcept"}


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_csv(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8-sig") as f:
        return [_clean_row(row) for row in csv.DictReader(f)]


def _clean_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _clean_row(row: dict) -> dict:
    return {
        _clean_text(key): _clean_text(value)
        for key, value in row.items()
        if key is not None
    }


def _dedupe_rows(rows: list[dict], keys: list[str]) -> list[dict]:
    seen = set()
    result = []
    for row in rows:
        marker = tuple(row.get(key, "") for key in keys)
        if marker in seen:
            continue
        seen.add(marker)
        result.append(row)
    return result


def _clean_entities(rows: list[dict]) -> list[dict]:
    cleaned = _dedupe_rows(rows, ["id", "label", "name"])
    seen_ids = set()
    for row in cleaned:
        if not row.get("id") or not row.get("label") or not row.get("name"):
            raise ValueError("entities.csv contains rows with empty id, label, or name")
        if row["label"] not in VALID_ENTITY_LABELS:
            raise ValueError(f"Unsupported entity label: {row['label']}")
        if row["id"] in seen_ids:
            raise ValueError(f"Duplicate entity id: {row['id']}")
        seen_ids.add(row["id"])
    return cleaned


def _clean_relations(rows: list[dict]) -> list[dict]:
    cleaned = _dedupe_rows(rows, ["start_id", "type", "end_id"])
    for row in cleaned:
        if not row.get("start_id") or not row.get("type") or not row.get("end_id"):
            raise ValueError("relations.csv contains rows with empty start_id, type, or end_id")
    return cleaned


def _entity_file_path(data_dir: str) -> Optional[str]:
    for filename in ("entities_final.csv", "entities.csv"):
        path = os.path.join(data_dir, filename)
        if os.path.exists(path):
            return path
    return None


def _relation_file_path(data_dir: str) -> Optional[str]:
    for filename in ("relations_final.csv", "relations.csv"):
        path = os.path.join(data_dir, filename)
        if os.path.exists(path):
            return path
    return None


def _read_excel(path: str) -> list[dict]:
    import openpyxl
    wb = openpyxl.load_workbook(path)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    headers = [str(h) for h in rows[0]]
    return [dict(zip(headers, row)) for row in rows[1:]]


def _parse_file(path: str) -> Any:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        return _read_json(path)
    elif ext == ".csv":
        return _read_csv(path)
    elif ext in (".xlsx", ".xls"):
        return _read_excel(path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def parse_data_dir(data_dir: str) -> dict:
    """解析数据目录中的所有文件，返回统一格式的 dict"""
    entity_path = _entity_file_path(data_dir)
    relation_path = _relation_file_path(data_dir)
    graph_paths = {
        "entities": entity_path,
        "relations": relation_path,
        "knowledge_concept_audit": os.path.join(data_dir, "knowledge_concept_audit.csv"),
    }
    if graph_paths["entities"]:
        result = {
            "format": "entity_relation_csv",
            "entities": _clean_entities(_read_csv(graph_paths["entities"])),
            "relations": [],
        }
        if graph_paths["relations"]:
            result["relations"] = _clean_relations(_read_csv(graph_paths["relations"]))
        if os.path.exists(graph_paths["knowledge_concept_audit"]):
            result["knowledge_concept_audit"] = _read_csv(graph_paths["knowledge_concept_audit"])
        return result

    result = {}

    expected_files = {
        "majors": ["majors.json", "majors.csv", "majors.xlsx"],
        "courses": ["courses.json", "courses.csv", "courses.xlsx"],
        "knowledge": ["knowledge.json", "knowledge.csv", "knowledge.xlsx"],
        "major_course_rels": ["major_course_rels.json"],
        "course_knowledge_rels": ["course_knowledge_rels.json"],
        "knowledge_prereq_rels": ["knowledge_prereq_rels.json"],
    }

    for key, filenames in expected_files.items():
        for fname in filenames:
            path = os.path.join(data_dir, fname)
            if os.path.exists(path):
                result[key] = _parse_file(path)
                break
        if key not in result:
            raise FileNotFoundError(f"Missing data file for '{key}' in {data_dir}")

    optional_files = {
        "course_prereq_rels": ["course_prereq_rels.json"],
    }
    for key, filenames in optional_files.items():
        for fname in filenames:
            path = os.path.join(data_dir, fname)
            if os.path.exists(path):
                result[key] = _parse_file(path)
                break

    return result
