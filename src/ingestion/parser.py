"""多格式数据解析器：支持 JSON/CSV/Excel"""
import json
import csv
import os
from typing import Any


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_csv(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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

    return result
