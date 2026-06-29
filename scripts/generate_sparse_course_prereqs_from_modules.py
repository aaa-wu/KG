"""从模块依赖图推导稀疏的课程间前置关系.

对于每个模块依赖 A depends_on B，从 B（前置模块）和 A（后续模块）中
各选最多 MAX_PER_MODULE 门代表性课程，生成 PREREQUISITE_FOR 边.

输出:
- data/course_prereq_from_modules_sparse.csv

用法:
    python3 scripts/generate_sparse_course_prereqs_from_modules.py
"""

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.recommendation.roadmap_classifier import ROADMAP_MODULES

MAX_PER_MODULE = 3  # 每个模块最多选几门课


def _normalize(name: str) -> str:
    return name.strip().lower()


def _match_course_to_module(course_name: str, course_to_module: dict[str, str], fallback_module: str) -> str:
    name = _normalize(course_name)
    best_module = fallback_module
    best_len = 0
    for keyword, module_id in course_to_module.items():
        kw = _normalize(keyword)
        if not kw or kw not in name:
            continue
        if len(kw) > best_len:
            best_len = len(kw)
            best_module = module_id
    return best_module


def load_courses(csv_path: Path) -> dict[str, str]:
    courses = {}
    with csv_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("label") == "Course":
                courses[row["id"]] = row["name"]
    return courses


def load_major_name_by_id(entities_path: Path) -> dict[str, str]:
    major_map = {}
    with entities_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("label") == "Major":
                major_map[row["id"]] = row["name"]
    return major_map


def load_course_to_majors(relations_path: Path, major_map: dict[str, str]) -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = {}
    with relations_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("type") == "HAS_COURSE":
                major_id = row.get("start_id", "")
                course_id = row.get("end_id", "")
                major_name = major_map.get(major_id)
                if major_name and course_id:
                    mapping.setdefault(course_id, set()).add(major_name)
    return mapping


def main() -> None:
    data_dir = PROJECT_ROOT / "data"
    entities_path = data_dir / "entities_final.csv"
    relations_path = data_dir / "relations_final.csv"
    output_path = data_dir / "course_prereq_from_modules_sparse.csv"

    all_courses = load_courses(entities_path)
    major_map = load_major_name_by_id(entities_path)
    course_to_majors = load_course_to_majors(relations_path, major_map)

    edges: set[tuple[str, str]] = set()

    for major_name, config in ROADMAP_MODULES.items():
        modules_config = config["modules"]
        course_to_module = config["course_to_module"]
        fallback_module = config.get("fallback_module", modules_config[0]["id"])
        module_ids = {m["id"] for m in modules_config}

        # 把属于该专业的课程分配到模块
        module_courses: dict[str, list[str]] = {m["id"]: [] for m in modules_config}
        for cid, cname in all_courses.items():
            if major_name not in course_to_majors.get(cid, set()):
                continue
            module_id = _match_course_to_module(cname, course_to_module, fallback_module)
            if module_id in module_ids:
                module_courses[module_id].append(cid)

        # 对每个模块依赖，稀疏生成课程前置边
        for m in modules_config:
            target_module = m["id"]
            for source_module in m.get("depends_on", []):
                if source_module not in module_ids or target_module not in module_ids:
                    continue

                source_courses = sorted(module_courses.get(source_module, []))[:MAX_PER_MODULE]
                target_courses = sorted(module_courses.get(target_module, []))[:MAX_PER_MODULE]

                for source_cid in source_courses:
                    for target_cid in target_courses:
                        if source_cid == target_cid:
                            continue
                        edges.add((source_cid, target_cid))

    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["start_id", "type", "end_id"])
        for source_id, target_id in sorted(edges):
            writer.writerow([source_id, "PREREQUISITE_FOR", target_id])

    print(f"Generated {len(edges)} sparse course prerequisite edges -> {output_path}")


if __name__ == "__main__":
    main()
