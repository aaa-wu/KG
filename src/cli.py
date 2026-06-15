"""命令行入口：交互对话 + 数据导入 + 专业对比"""
import sys
import json
from typing import Optional

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
from src.ontology.audit import audit_graph
from src.ontology.topic_extractor import extract_topics_for_major_from_neo4j
from src.ontology.validator import (
    queue_extraction_result,
    get_pending_validations,
    approve_extraction,
    reject_extraction,
    load_validated_items,
)
from src.ontology.importer import import_validated_topics
from src.ontology.similarity import compute_and_store_similarity
from src.prereq_prediction.predictor import predict_and_store
from src.prereq_prediction.llm_predictor import complete_prerequisites_with_llm, infer_real_prerequisites_with_llm
from src.rl.recommend import recommend_path


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
    if data.get("format") == "entity_relation_csv":
        labels = {}
        for row in data.get("entities", []):
            labels[row["label"]] = labels.get(row["label"], 0) + 1
        rels = {}
        for row in data.get("relations", []):
            rels[row["type"]] = rels.get(row["type"], 0) + 1
        print(f"  实体: {len(data.get('entities', []))} {labels}")
        print(f"  关系: {len(data.get('relations', []))} {rels}")
        print(f"  审计记录: {len(data.get('knowledge_concept_audit', []))}")
    else:
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
    print(f"  课程前驱关系: {counts['course_prereq_rels']}")
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


def cmd_audit(output_path: str = "data/audit_report.json"):
    """审计图谱数据质量"""
    report = audit_graph(output_path)
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))


def cmd_extract_topics(major_name: str, queue: bool = False):
    """使用 DeepSeek 从现有课程/知识点名中抽取 Topic/SubTopic"""
    result = extract_topics_for_major_from_neo4j(major_name)
    if result is None:
        print("LLM 抽取不可用（未配置 DEEPSEEK_API_KEY 或调用失败）")
        return
    if queue:
        queued_ids = queue_extraction_result(result)
        print(f"已为专业 {major_name} 生成 {len(queued_ids)} 个待审 Topic 提案")
        return
    print(json.dumps({
        "major": result.major,
        "topics": [
            {"name": t.name, "domain": t.domain, "subtopics": t.subtopics}
            for t in result.topics
        ],
    }, ensure_ascii=False, indent=2))


def cmd_validate(action: str = "list", item_id: str = "", notes: str = ""):
    """人工复核 Topic/SubTopic 抽取结果"""
    if action == "list":
        items = get_pending_validations(50)
        print(f"待审项目 ({len(items)} 个):")
        for item in items:
            print(f"  {item.id}: {item.major} / {item.domain} / {item.topic}")
            print(f"    subtopics: {', '.join(item.subtopics[:5])}{'...' if len(item.subtopics) > 5 else ''}")
    elif action == "approve":
        if not item_id:
            print("请使用 --id 指定待审项 ID")
            return
        if approve_extraction(item_id, notes):
            validated = load_validated_items()
            counts = import_validated_topics(validated)
            print(f"已批准 {item_id} 并导入。当前导入统计: {counts}")
        else:
            print("待审项不存在或已处理")
    elif action == "reject":
        if not item_id:
            print("请使用 --id 指定待审项 ID")
            return
        if reject_extraction(item_id, notes):
            print(f"已拒绝 {item_id}")
        else:
            print("待审项不存在或已处理")
    else:
        print("未知操作: list | approve | reject")


def cmd_similarity(entity_label: str = "KnowledgeConcept", threshold: float = 0.75):
    """计算并存储语义相似度边"""
    result = compute_and_store_similarity(entity_label=entity_label, threshold=threshold)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_predict_prereq(
    threshold: float = 0.7,
    dry_run: bool = True,
    max_predictions: int = 500,
    method: str = "llm",
    max_targets: Optional[int] = None,
    top_k: int = 15,
):
    """预测缺失的前置知识点关系"""
    if method == "llm":
        result = complete_prerequisites_with_llm(
            top_k=top_k,
            min_score=threshold,
            max_targets=max_targets,
            dry_run=dry_run,
        )
    else:
        result = predict_and_store(
            threshold=threshold,
            dry_run=dry_run,
            max_predictions=max_predictions,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_infer_prereq(
    threshold: float = 0.75,
    dry_run: bool = True,
    max_targets: Optional[int] = None,
    top_k: int = 15,
):
    """从课程结构推断真实前置知识点关系并写入 CONCEPT_PREREQUISITE_FOR。"""
    result = infer_real_prerequisites_with_llm(
        top_k=top_k,
        min_score=threshold,
        max_targets=max_targets,
        dry_run=dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


# Common patterns that the LLM predictor confuses due to surface word similarity.
# Each entry is a pair (source_keyword, target_keyword). Predicted edges where the
# source name contains source_keyword and the target name contains target_keyword
# are considered bogus and removed during cleanup.
BOGUS_PREDICTED_PATTERNS = [
    # Artificial neural networks vs biological neuroscience
    ("神经元", "神经网络"),
    ("神经科学", "神经网络"),
    ("神经网络", "神经科学"),
    ("神经元", "神经科学"),
    ("神经科学", "神经元"),
    ("心理科学", "神经科学"),
    ("心理科学", "神经网络"),
    ("认知心理学", "神经网络"),
    ("行为神经科学", "神经网络"),
]


def cmd_clean_predicted_prereq(
    min_confidence: Optional[float] = None,
    remove_all: bool = False,
    dry_run: bool = True,
) -> dict:
    """清理低质量或明显错误的 PREDICTED_PREREQUISITE 边。"""
    from src.models.schema import LABEL_KNOWLEDGE_POINT, REL_PREDICTED_PREREQ

    driver = get_neo4j_driver()
    removed = 0
    with driver.session() as session:
        if remove_all:
            result = session.run(
                f"""
                MATCH (a:{LABEL_KNOWLEDGE_POINT})-[r:{REL_PREDICTED_PREREQ}]->(b:{LABEL_KNOWLEDGE_POINT})
                RETURN count(r) AS cnt
                """
            )
            cnt = result.single()["cnt"]
            if not dry_run:
                session.run(
                    f"""
                    MATCH (a:{LABEL_KNOWLEDGE_POINT})-[r:{REL_PREDICTED_PREREQ}]->(b:{LABEL_KNOWLEDGE_POINT})
                    DELETE r
                    """
                )
            removed = cnt
        else:
            # Remove edges below confidence threshold
            if min_confidence is not None:
                result = session.run(
                    f"""
                    MATCH (a:{LABEL_KNOWLEDGE_POINT})-[r:{REL_PREDICTED_PREREQ}]->(b:{LABEL_KNOWLEDGE_POINT})
                    WHERE r.confidence IS NULL OR r.confidence < $min_confidence
                    RETURN count(r) AS cnt
                    """,
                    min_confidence=min_confidence,
                )
                cnt = result.single()["cnt"]
                if not dry_run:
                    session.run(
                        f"""
                        MATCH (a:{LABEL_KNOWLEDGE_POINT})-[r:{REL_PREDICTED_PREREQ}]->(b:{LABEL_KNOWLEDGE_POINT})
                        WHERE r.confidence IS NULL OR r.confidence < $min_confidence
                        DELETE r
                        """,
                        min_confidence=min_confidence,
                    )
                removed += cnt

            # Remove bogus cross-domain edges
            for src_kw, dst_kw in BOGUS_PREDICTED_PATTERNS:
                result = session.run(
                    f"""
                    MATCH (a:{LABEL_KNOWLEDGE_POINT})-[r:{REL_PREDICTED_PREREQ}]->(b:{LABEL_KNOWLEDGE_POINT})
                    WHERE a.name CONTAINS $src_kw AND b.name CONTAINS $dst_kw
                    RETURN count(r) AS cnt
                    """,
                    src_kw=src_kw,
                    dst_kw=dst_kw,
                )
                cnt = result.single()["cnt"]
                if not dry_run:
                    session.run(
                        f"""
                        MATCH (a:{LABEL_KNOWLEDGE_POINT})-[r:{REL_PREDICTED_PREREQ}]->(b:{LABEL_KNOWLEDGE_POINT})
                        WHERE a.name CONTAINS $src_kw AND b.name CONTAINS $dst_kw
                        DELETE r
                        """,
                        src_kw=src_kw,
                        dst_kw=dst_kw,
                    )
                removed += cnt

    status = "dry_run" if dry_run else "success"
    report = {
        "status": status,
        "removed_count": removed,
        "min_confidence": min_confidence,
        "remove_all": remove_all,
        "dry_run": dry_run,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def cmd_recommend(target: str, known: str = "", level: str = "intermediate"):
    """使用双图 RL 推荐学习路径"""
    known_list = [k.strip() for k in known.split(",") if k.strip()] if known else []
    result = recommend_path(
        known_concepts=known_list,
        target_concept=target,
        level=level,
    )
    print(json.dumps({
        "target": result.target,
        "explanation": result.explanation,
        "path": result.path,
        "blocked_recovery": result.blocked_recovery,
        "difficulty_analysis": result.difficulty_analysis,
    }, ensure_ascii=False, indent=2))


def print_usage():
    print("""用法:
  python3 -m src.cli chat                    交互式对话模式
  python3 -m src.cli init                    初始化 Neo4j Schema
  python3 -m src.cli import --dir DIR        导入数据
  python3 -m src.cli audit [--out PATH]      审计图谱数据质量
  python3 -m src.cli extract-topics --major M [--queue]  抽取 Topic/SubTopic 提案
  python3 -m src.cli validate [list|approve|reject] [--id ID] [--notes TEXT]  复核队列
  python3 -m src.cli similarity [--label LABEL] [--threshold T]  计算语义相似度
  python3 -m src.cli predict-prereq [--method llm|mlp] [--threshold T] [--dry-run false] [--max-targets N] [--top-k K]  前置预测（写入 PREDICTED_PREREQUISITE）
  python3 -m src.cli infer-prereq [--threshold T] [--dry-run false] [--max-targets N] [--top-k K]  推断真实前置（写入 CONCEPT_PREREQUISITE_FOR）
  python3 -m src.cli clean-predicted [--min-confidence C] [--all] [--dry-run false]  清理低质量预测前置边
  python3 -m src.cli recommend --target NAME [--known "A,B"] [--level L]  双图推荐
  python3 -m src.cli path --target NAME [--known "A,B"]  查找知识点学习路径
  python3 -m src.cli compare --major M --uni-a A --uni-b B  对比专业

说明:
  path --target 当前应传入知识点名称；课程/专业路径规划请使用 chat 或 recommend。
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
        data_dir = "data"
        for i, a in enumerate(args):
            if a == "--dir" and i + 1 < len(args):
                data_dir = args[i + 1]
        cmd_import(data_dir)
    elif cmd == "audit":
        output_path = "data/audit_report.json"
        for i, a in enumerate(args):
            if a == "--out" and i + 1 < len(args):
                output_path = args[i + 1]
        cmd_audit(output_path)
    elif cmd == "extract-topics":
        major = ""
        queue = False
        for i, a in enumerate(args):
            if a == "--major" and i + 1 < len(args):
                major = args[i + 1]
            if a == "--queue":
                queue = True
        if not major:
            print("请使用 --major 指定专业")
            return
        cmd_extract_topics(major, queue=queue)
    elif cmd == "validate":
        action = "list"
        item_id = ""
        notes = ""
        for i, a in enumerate(args):
            if a in ("list", "approve", "reject"):
                action = a
            if a == "--id" and i + 1 < len(args):
                item_id = args[i + 1]
            if a == "--notes" and i + 1 < len(args):
                notes = args[i + 1]
        cmd_validate(action, item_id, notes)
    elif cmd == "similarity":
        entity_label = "KnowledgeConcept"
        threshold = 0.75
        for i, a in enumerate(args):
            if a == "--label" and i + 1 < len(args):
                entity_label = args[i + 1]
            if a == "--threshold" and i + 1 < len(args):
                threshold = float(args[i + 1])
        cmd_similarity(entity_label, threshold)
    elif cmd == "predict-prereq":
        threshold = 0.7
        dry_run = True
        max_predictions = 500
        method = "llm"
        max_targets = None
        top_k = 15
        for i, a in enumerate(args):
            if a == "--threshold" and i + 1 < len(args):
                threshold = float(args[i + 1])
            if a == "--dry-run" and i + 1 < len(args):
                dry_run = args[i + 1].lower() in ("true", "1", "yes")
            if a == "--max" and i + 1 < len(args):
                max_predictions = int(args[i + 1])
            if a == "--method" and i + 1 < len(args):
                method = args[i + 1]
            if a == "--max-targets" and i + 1 < len(args):
                max_targets = int(args[i + 1])
            if a == "--top-k" and i + 1 < len(args):
                top_k = int(args[i + 1])
        cmd_predict_prereq(threshold, dry_run, max_predictions, method, max_targets, top_k)
    elif cmd == "infer-prereq":
        threshold = 0.75
        dry_run = True
        max_targets = None
        top_k = 15
        for i, a in enumerate(args):
            if a == "--threshold" and i + 1 < len(args):
                threshold = float(args[i + 1])
            if a == "--dry-run" and i + 1 < len(args):
                dry_run = args[i + 1].lower() in ("true", "1", "yes")
            if a == "--max-targets" and i + 1 < len(args):
                max_targets = int(args[i + 1])
            if a == "--top-k" and i + 1 < len(args):
                top_k = int(args[i + 1])
        cmd_infer_prereq(threshold, dry_run, max_targets, top_k)
    elif cmd == "clean-predicted":
        min_confidence = None
        remove_all = False
        dry_run = True
        for i, a in enumerate(args):
            if a == "--min-confidence" and i + 1 < len(args):
                min_confidence = float(args[i + 1])
            if a == "--all":
                remove_all = True
            if a == "--dry-run" and i + 1 < len(args):
                dry_run = args[i + 1].lower() in ("true", "1", "yes")
        cmd_clean_predicted_prereq(min_confidence, remove_all, dry_run)
    elif cmd == "recommend":
        target = ""
        known = ""
        level = "intermediate"
        for i, a in enumerate(args):
            if a == "--target" and i + 1 < len(args):
                target = args[i + 1]
            if a == "--known" and i + 1 < len(args):
                known = args[i + 1]
            if a == "--level" and i + 1 < len(args):
                level = args[i + 1]
        if not target:
            print("请使用 --target 指定目标")
            return
        cmd_recommend(target, known, level)
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
