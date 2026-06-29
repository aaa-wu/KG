"""从模块依赖图推导中等密度的课程间前置关系，并导出为 CSV.

策略:
1. 语义优先: 对每条模块依赖 A -> B，若 A 中某课程覆盖的知识点包含 B 中某课程
   所需知识点的前置知识点，则生成 PREREQUISITE_FOR 边，并按前置知识点
   重叠数保留前 SEMANTIC_K 条.
2. 动态补充: 若某条模块依赖没有语义边，则按模块大小动态选取代表性课程
   (小模块 3 门、中模块 5 门、大模块 6–8 门) 进行补边，避免过度稀疏.

输出:
    data/course_prereq_from_modules.csv
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.recommendation.roadmap_classifier import ROADMAP_MODULES

SEMANTIC_K = 10  # 每门后续课程最多保留几条语义前置边


def _normalize(name: str) -> str:
    """与 roadmap_classifier.py 一致的归一化：小写并去除首尾空格."""
    return name.strip().lower()


def _match_course_to_module(course_name: str, course_to_module: dict[str, str], fallback_module: str) -> str:
    """复制 roadmap_classifier.py 的最长子串匹配逻辑."""
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
    """返回 {course_id: course_name}."""
    courses = {}
    with csv_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("label") == "Course":
                courses[row["id"]] = row["name"]
    return courses


def load_course_to_majors(relations_path: Path) -> dict[str, set[str]]:
    """从 HAS_COURSE 关系推导课程所属专业."""
    major_map = {}
    entities_path = relations_path.parent / "entities_final.csv"
    with entities_path.open("r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("label") == "Major":
                major_map[row["id"]] = row["name"]

    mapping: dict[str, set[str]] = {}
    with relations_path.open("r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("type") == "HAS_COURSE":
                major_name = major_map.get(row.get("start_id", ""))
                course_id = row.get("end_id", "")
                if major_name and course_id:
                    mapping.setdefault(course_id, set()).add(major_name)
    return mapping


def derive_course_prereqs(
    all_courses: dict[str, str],
    course_to_majors: dict[str, set[str]],
    covers: dict[str, set[str]],
    concept_prereqs: dict[str, set[str]],
) -> set[tuple[str, str]]:
    """基于模块依赖与知识点前置链推导课程前置关系."""
    edges: set[tuple[str, str]] = set()

    for major_name, config in ROADMAP_MODULES.items():
        modules_config = config["modules"]
        course_to_module = config["course_to_module"]
        fallback_module = config.get("fallback_module", modules_config[0]["id"])
        module_ids = {m["id"] for m in modules_config}

        # 仅把属于当前专业的课程分配到该专业的模块
        module_courses: dict[str, list[str]] = {m["id"]: [] for m in modules_config}
        for cid, cname in all_courses.items():
            if major_name not in course_to_majors.get(cid, set()):
                continue
            module_id = _match_course_to_module(cname, course_to_module, fallback_module)
            if module_id not in module_courses:
                module_id = fallback_module
            module_courses[module_id].append(cid)

        # 按模块大小动态决定代表课程数
        def reps(module_id: str) -> list[str]:
            cids = module_courses.get(module_id, [])
            size = len(cids)
            if size <= 10:
                n = 3
            elif size <= 30:
                n = 5
            elif size <= 60:
                n = 6
            else:
                n = 8
            return sorted(cids, key=lambda c: (-len(covers.get(c, set())), c))[:n]

        # 遍历模块依赖生成课程前置边
        for m in modules_config:
            target_module = m["id"]
            for source_module in m.get("depends_on", []):
                if source_module not in module_ids or target_module not in module_ids:
                    continue

                source_courses = module_courses.get(source_module, [])
                target_courses = module_courses.get(target_module, [])
                if not source_courses or not target_courses:
                    continue

                dep_edges: set[tuple[str, str]] = set()

                # 1) 语义优先：基于知识点前置链
                for target_cid in target_courses:
                    target_concepts = covers.get(target_cid, set())
                    needed_prereqs: set[str] = set()
                    for c in target_concepts:
                        needed_prereqs.update(concept_prereqs.get(c, set()))
                    if not needed_prereqs:
                        continue

                    scored = []
                    for source_cid in source_courses:
                        if source_cid == target_cid:
                            continue
                        overlap = len(covers.get(source_cid, set()) & needed_prereqs)
                        if overlap > 0:
                            scored.append((overlap, source_cid))

                    scored.sort(reverse=True)
                    for _, source_cid in scored[:SEMANTIC_K]:
                        dep_edges.add((source_cid, target_cid))

                # 2) 动态补充：没有语义边时，用代表课程补边
                if not dep_edges:
                    for source_cid in reps(source_module):
                        for target_cid in reps(target_module):
                            if source_cid != target_cid:
                                dep_edges.add((source_cid, target_cid))

                edges.update(dep_edges)

    return edges


def main() -> None:
    data_dir = PROJECT_ROOT / "data"
    entities_path = data_dir / "entities_final.csv"
    relations_path = data_dir / "relations_final.csv"
    output_path = data_dir / "course_prereq_from_modules.csv"

    all_courses = load_courses(entities_path)
    print(f"Loaded {len(all_courses)} courses from {entities_path}")

    course_to_majors = load_course_to_majors(relations_path)

    # 加载课程-知识点覆盖关系
    covers: dict[str, set[str]] = defaultdict(set)
    concept_prereqs: dict[str, set[str]] = defaultdict(set)
    with relations_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("type") == "COVERS_KNOWLEDGE":
                covers[row["start_id"]].add(row["end_id"])
            elif row.get("type") == "CONCEPT_PREREQUISITE_FOR":
                concept_prereqs[row["end_id"]].add(row["start_id"])

    edges = derive_course_prereqs(all_courses, course_to_majors, covers, concept_prereqs)
    print(f"Derived {len(edges)} course-to-course prerequisite edges from module dependencies")

    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["start_id", "type", "end_id"])
        for source_id, target_id in sorted(edges):
            writer.writerow([source_id, "PREREQUISITE_FOR", target_id])

    print(f"Written to {output_path}")


if __name__ == "__main__":
    main()
