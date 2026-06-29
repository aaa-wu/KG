"""从 roadmap_classifier.py 导出模块依赖图数据为 CSV.

生成三个文件:
- data/module_nodes.csv: 模块节点 (id, label, name, major, level)
- data/module_dependencies.csv: 模块间依赖 (start_id, type, end_id)
- data/course_to_module.csv: 课程归属模块 (start_id, type, end_id)

用法:
    python3 scripts/export_module_dependency_graph.py
"""

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.recommendation.roadmap_classifier import ROADMAP_MODULES


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
    """返回 {major_id: major_name}."""
    major_map = {}
    with entities_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("label") == "Major":
                major_map[row["id"]] = row["name"]
    return major_map


def load_course_to_majors(relations_path: Path, major_map: dict[str, str]) -> dict[str, set[str]]:
    """返回 {course_id: {major_name, ...}}."""
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


def compute_module_levels(modules_config: list[dict]) -> dict[str, int]:
    """复制 roadmap_classifier.py 的模块层级计算."""
    module_ids = {m["id"] for m in modules_config}
    levels: dict[str, int] = {m["id"]: 0 for m in modules_config}

    def level_of(mid: str) -> int:
        if mid not in levels:
            return 0
        if levels[mid] != 0:
            return levels[mid]
        m_config = next((m for m in modules_config if m["id"] == mid), None)
        if not m_config or not m_config.get("depends_on"):
            levels[mid] = 0
            return 0
        deps = [d for d in m_config["depends_on"] if d in module_ids]
        if not deps:
            levels[mid] = 0
            return 0
        levels[mid] = max(level_of(d) for d in deps) + 1
        return levels[mid]

    for m in modules_config:
        level_of(m["id"])
    return levels


def main() -> None:
    data_dir = PROJECT_ROOT / "data"
    entities_path = data_dir / "entities_final.csv"
    relations_path = data_dir / "relations_final.csv"

    module_nodes_path = data_dir / "module_nodes.csv"
    module_deps_path = data_dir / "module_dependencies.csv"
    course_to_module_path = data_dir / "course_to_module.csv"

    all_courses = load_courses(entities_path)
    major_map = load_major_name_by_id(entities_path)
    course_to_majors = load_course_to_majors(relations_path, major_map)
    print(f"Loaded {len(all_courses)} courses across {len(major_map)} majors")

    module_rows: list[dict] = []
    module_dep_rows: list[tuple[str, str, str]] = []
    course_module_rows: list[tuple[str, str, str]] = []

    for major_name, config in ROADMAP_MODULES.items():
        modules_config = config["modules"]
        course_to_module = config["course_to_module"]
        fallback_module = config.get("fallback_module", modules_config[0]["id"])
        module_ids = {m["id"] for m in modules_config}
        levels = compute_module_levels(modules_config)

        # 模块节点
        for m in modules_config:
            module_rows.append({
                "id": f"M_{major_name}_{m['id']}",
                "label": "Module",
                "name": f"{major_name}·{m['name']}",
                "major": major_name,
                "level": levels.get(m["id"], 0),
            })

        # 模块间依赖
        for m in modules_config:
            target_id = f"M_{major_name}_{m['id']}"
            for dep_id in m.get("depends_on", []):
                if dep_id in module_ids:
                    source_id = f"M_{major_name}_{dep_id}"
                    module_dep_rows.append((source_id, "MODULE_PREREQUISITE_FOR", target_id))

        # 课程归属模块：只把属于该专业的课程归到该专业的模块
        for cid, cname in all_courses.items():
            if major_name not in course_to_majors.get(cid, set()):
                continue
            module_id = _match_course_to_module(cname, course_to_module, fallback_module)
            if module_id in module_ids:
                course_module_rows.append((cid, "BELONGS_TO_MODULE", f"M_{major_name}_{module_id}"))

    # 写 module_nodes.csv
    with module_nodes_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label", "name", "major", "level"])
        writer.writeheader()
        writer.writerows(module_rows)

    # 写 module_dependencies.csv
    with module_deps_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["start_id", "type", "end_id"])
        writer.writerows(sorted(set(module_dep_rows)))

    # 写 course_to_module.csv
    with course_to_module_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["start_id", "type", "end_id"])
        writer.writerows(sorted(set(course_module_rows)))

    print(f"Generated {len(module_rows)} module nodes -> {module_nodes_path}")
    print(f"Generated {len(set(module_dep_rows))} module dependencies -> {module_deps_path}")
    print(f"Generated {len(set(course_module_rows))} course-to-module mappings -> {course_to_module_path}")


if __name__ == "__main__":
    main()
