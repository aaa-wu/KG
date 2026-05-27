"""命令行入口：交互对话 + 数据导入 + 专业对比"""
import sys
import json

from src.config import get_neo4j_driver
from src.models.schema import init_schema
from src.ingestion.parser import parse_data_dir
from src.ingestion.importer import import_all
from src.recommendation.path_finder import (
    find_prerequisites,
    recommend_next,
    find_path_to_target,
    get_major_structure,
)
from src.recommendation.compare import compare_majors
from src.llm.agent import (
    resolve_target_and_known,
    format_plan,
    get_all_knowledge_point_names,
    get_all_major_names,
)


def cmd_init():
    """初始化 Neo4j Schema（约束和索引）"""
    driver = get_neo4j_driver()
    count = init_schema(driver)
    print(f"Schema 初始化完成，执行了 {count} 条语句")
    driver.close()


def cmd_import(data_dir: str):
    """导入数据目录中的所有文件到 Neo4j"""
    print(f"解析数据目录: {data_dir}")
    data = parse_data_dir(data_dir)
    print(f"  专业: {len(data.get('majors', []))}")
    print(f"  课程: {len(data.get('courses', []))}")
    print(f"  知识点: {len(data.get('knowledge', []))}")

    driver = get_neo4j_driver()
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        counts = import_all(session, data)

    print("\n导入完成:")
    print(f"  专业节点: {counts['majors']}")
    print(f"  课程节点: {counts['courses']}")
    print(f"  知识点节点: {counts['knowledge_points']}")
    print(f"  专业-课程关系: {counts['major_course_rels']}")
    print(f"  课程-知识点关系: {counts['course_knowledge_rels']}")
    print(f"  前驱关系: {counts['knowledge_prereq_rels']}")
    driver.close()


def cmd_path(target: str, known: str = ""):
    """查找学习路径"""
    known_list = [k.strip() for k in known.split(",") if k.strip()] if known else []
    driver = get_neo4j_driver()
    with driver.session() as session:
        if known_list:
            result = find_path_to_target(session, known_list, target)
        else:
            result = find_prerequisites(session, target)

    if "error" in result:
        print(f"错误: {result['error']}")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    driver.close()


def cmd_compare(major: str, uni_a: str, uni_b: str):
    """对比两校专业"""
    driver = get_neo4j_driver()
    with driver.session() as session:
        result = compare_majors(session, major, uni_a, uni_b)

    print(f"\n===== {major} 专业对比: {uni_a} vs {uni_b} =====")
    print(f"课程数: {uni_a} {result['course_count']['a']} | {uni_b} {result['course_count']['b']}")
    print(f"知识点数: {uni_a} {result['kp_count']['a']} | {uni_b} {result['kp_count']['b']}")
    print(f"课程重叠率: {result['course_overlap_rate']}")
    print(f"知识点重叠率: {result['kp_overlap_rate']}")

    if result["only_in_a"]["courses"]:
        print(f"\n{uni_a} 独有的课程: {', '.join(result['only_in_a']['courses'])}")
    if result["only_in_b"]["courses"]:
        print(f"{uni_b} 独有的课程: {', '.join(result['only_in_b']['courses'])}")

    if result["only_in_a"]["knowledge_points"]:
        print(f"{uni_a} 独有的知识点 ({len(result['only_in_a']['knowledge_points'])}个): "
              f"{', '.join(result['only_in_a']['knowledge_points'][:10])}...")
    if result["only_in_b"]["knowledge_points"]:
        print(f"{uni_b} 独有的知识点 ({len(result['only_in_b']['knowledge_points'])}个): "
              f"{', '.join(result['only_in_b']['knowledge_points'][:10])}...")
    driver.close()


def _run_recommendation(session, resolved: dict) -> dict:
    """根据解析后的意图执行图查询"""
    action = resolved["action"]
    target = resolved["target"]
    target_type = resolved.get("target_type")
    known = resolved["known"]
    university = resolved.get("university")

    if action == "recommend_path":
        if target_type == "Major":
            return get_major_structure(session, target, university)
        elif target_type == "Course":
            result = get_major_structure(session, None, None)
            # 过滤出目标课程及其前驱课程的知识点
            target_courses = []
            for c in result.get("courses", []):
                if c["name"] == target:
                    target_courses.append(c)
            # 获取课程知识点前驱链
            if target_courses:
                kps = target_courses[0].get("knowledge_points", [])
                if kps:
                    return find_prerequisites(session, kps[-1])
            return {"target": target, "courses": target_courses}
        else:
            if known:
                return find_path_to_target(session, known, target)
            else:
                return find_prerequisites(session, target)

    elif action == "compare_majors":
        all_majors = get_all_major_names()
        if len(all_majors) >= 2:
            # 默认对比前两个不同学校的同名专业
            import itertools
            for a, b in itertools.combinations(all_majors, 2):
                result = compare_majors(session, a, "", "")
                return result
        return {"error": "Not enough majors to compare"}

    elif action == "explore_major":
        return get_major_structure(session, target, university)

    elif action == "unknown":
        return {"error": "无法理解你的意图，请尝试更具体地描述你想学什么或对比什么"}

    return {"error": f"Unknown action: {action}"}


def cmd_chat():
    """交互式对话模式"""
    print("=" * 60)
    print("  知识图谱学习路径推荐系统")
    print("  输入 'quit' 或 'exit' 退出")
    print("  输入目标专业/知识点，我会为你规划学习路径")
    print("=" * 60)

    driver = get_neo4j_driver()

    while True:
        try:
            user_input = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("再见！")
            break

        print("思考中...")

        # Step 1: 意图解析 + 实体匹配
        resolved = resolve_target_and_known(user_input)
        action = resolved.get("action", "unknown")
        target = resolved.get("target", "")
        known = resolved.get("known", [])

        if action == "unknown":
            print("\n抱歉，我没有理解你的意图。请尝试：")
            print("  - '我想学[专业/知识点]'")
            print("  - '我会[知识点A, B]，想学[目标]'")
            print("  - '对比[学校A]和[学校B]的[专业]'")
            continue

        print(f"  [解析: action={action}, target={target}, known={known}]")

        # Step 2: 图查询
        with driver.session() as session:
            path_result = _run_recommendation(session, resolved)

        if "error" in path_result:
            print(f"\n{path_result['error']}")
            continue

        # Step 3: LLM 格式化输出
        explanation = format_plan(path_result)
        print(f"\n{explanation}")

    driver.close()


def print_usage():
    print("""用法:
  python3 -m src.cli chat                    交互式对话模式
  python3 -m src.cli init                    初始化 Neo4j Schema
  python3 -m src.cli import --dir DIR        导入数据
  python3 -m src.cli path --target NAME [--known "A,B"]  查找知识点学习路径
  python3 -m src.cli compare --major M --uni-a A --uni-b B  对比专业

说明:
  path --target 当前应传入知识点名称；课程/专业路径规划请使用 chat 模式。
""")


def main():
    args = sys.argv[1:]

    if not args:
        cmd_chat()
        return

    cmd = args[0]

    if cmd == "chat":
        cmd_chat()
    elif cmd == "init":
        cmd_init()
    elif cmd == "import":
        data_dir = "data/sample"
        for i, a in enumerate(args):
            if a == "--dir" and i + 1 < len(args):
                data_dir = args[i + 1]
        cmd_import(data_dir)
    elif cmd == "path":
        target = ""
        known = ""
        for i, a in enumerate(args):
            if a == "--target" and i + 1 < len(args):
                target = args[i + 1]
            if a == "--known" and i + 1 < len(args):
                known = args[i + 1]
        if not target:
            print("请使用 --target 指定目标")
            return
        cmd_path(target, known)
    elif cmd == "compare":
        major = ""
        uni_a = ""
        uni_b = ""
        for i, a in enumerate(args):
            if a == "--major" and i + 1 < len(args):
                major = args[i + 1]
            if a == "--uni-a" and i + 1 < len(args):
                uni_a = args[i + 1]
            if a == "--uni-b" and i + 1 < len(args):
                uni_b = args[i + 1]
        if not all([major, uni_a, uni_b]):
            print("请使用 --major --uni-a --uni-b 指定参数")
            return
        cmd_compare(major, uni_a, uni_b)
    elif cmd in ("help", "--help", "-h"):
        print_usage()
    else:
        print(f"未知命令: {cmd}")
        print_usage()


if __name__ == "__main__":
    main()
