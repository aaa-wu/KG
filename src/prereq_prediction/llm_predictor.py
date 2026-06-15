"""基于 DeepSeek LLM 的前置关系补全。

与 graph_mlp.py 不同，这里不训练模型，而是直接让 LLM 根据知识点名称判断
“候选列表中的哪些知识点是目标知识点的明确前置知识”。

候选生成优先使用课程结构（同课共现、前置课程、同专业课程），embedding
相似度仅作为数量不足时的兜底。这样可以避免纯 embedding 方案把名字
相似但语义不同的概念（例如「神经网络」和「神经科学」）混在一起。
"""
import json
import os
from typing import Optional

import numpy as np

from src.config import get_deepseek_client, DEEPSEEK_MODEL, get_neo4j_driver
from src.models.schema import LABEL_KNOWLEDGE_POINT, REL_PREREQUISITE_OF, REL_PREDICTED_PREREQ
from src.prereq_prediction.embedder import KnowledgeEmbedder


DEFAULT_TOP_K = 15
DEFAULT_BATCH_SIZE = 20  # 每批处理多少个目标知识点


# 常见由于字面相似导致的跨学科混淆模式。
# 每个条目是 (source_keyword, target_keyword)。如果边的源名包含 source_keyword、
# 目标名包含 target_keyword，则视为低质量边，直接丢弃。
BOGUS_EDGE_PATTERNS = [
    # 人工智能/机器学习 vs 生物神经科学
    ("神经元", "神经网络"),
    ("神经科学", "神经网络"),
    ("神经网络", "神经科学"),
    ("心理科学", "神经网络"),
    ("认知心理学", "神经网络"),
    ("行为神经科学", "神经网络"),
    # 生物神经科学内部：方向通常反了才会到这里；保留神经元->神经科学，但排除反向
    ("神经科学", "神经元"),
]


def _is_bogus_edge(src: str, dst: str) -> bool:
    for src_kw, dst_kw in BOGUS_EDGE_PATTERNS:
        if src_kw in src and dst_kw in dst:
            return True
    return False


LLM_PREREQ_PROMPT = """你是一位高等教育课程专家。请判断：对于目标知识点「{target}」，下面候选知识点中哪些是它明确的前置知识（即学习「{target}」之前通常需要先学的内容）。

候选知识点：
{candidates}

要求：
1. 只选择确实具有前置依赖关系的项目，不要选择仅仅是同领域、相似词或同一上级概念下的项目。
2. 如果某个候选是目标知识点的子概念、特例、应用场景或后续扩展，则不要选。
3. 必须严格区分不同学科：
   - 人工智能/机器学习中的「神经网络」「深度学习」「卷积神经网络」等，前置应该是数学（线性代数、概率论、优化）、编程、机器学习基础等，而不是生物学的「神经元」「神经科学」。
   - 生物/心理/医学中的「神经元」「神经科学」等，前置才是生物学、心理学、解剖学等。
4. 方向必须正确：如果 A 是 B 的前置，则输出 "A"，不要反向把 B 作为 A 的前置。
5. 宁缺毋滥：不确定时返回空列表，不要把相似概念误当前置。

反例（这些都不应该选）：
- 目标为「神经网络」时，不要把「神经元」「神经科学」「心理科学」作为前置。
- 目标为「神经科学」时，不要把「神经网络」作为前置。
- 目标为「深度学习」时，不要把「深度工作」作为前置。

返回 JSON 格式：
{{
  "prerequisites": ["候选名1", "候选名2"],
  "confidence": "high|medium|low",
  "reasoning": "简要说明选择依据，并说明为什么排除了相似但不构成前置的候选"
}}

只输出 JSON，不要 Markdown 代码块。"""


def _parse_llm_json(raw: str) -> Optional[dict]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _confidence_to_score(confidence: str) -> float:
    return {"high": 0.9, "medium": 0.75, "low": 0.6}.get(confidence, 0.7)


def _get_embedding_candidates(
    target: str,
    embeddings: dict[str, list[float]],
    exclude: set[str],
    top_k: int = DEFAULT_TOP_K,
) -> list[str]:
    """用 embedding cosine 找出目标知识点最相似的 Top-K 候选（排除自身和已有前置）。

    注意：embedding 相似度容易把名字相似但语义不同的概念聚在一起（例如
    「神经网络」和「神经科学」），所以本方法只作为课程结构候选的补充，
    不做主要候选源。
    """
    if target not in embeddings:
        return []

    target_vec = np.array(embeddings[target], dtype=np.float32)
    target_norm = np.linalg.norm(target_vec)
    if target_norm == 0:
        return []

    scores = []
    for name, vec in embeddings.items():
        if name == target or name in exclude:
            continue
        v = np.array(vec, dtype=np.float32)
        norm = np.linalg.norm(v)
        if norm == 0:
            continue
        sim = float(np.dot(target_vec, v) / (target_norm * norm))
        scores.append((name, sim))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _ in scores[:top_k]]


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return None
    return float(np.dot(a, b) / (norm_a * norm_b))


def _filter_outliers(
    target: str,
    candidates: list[str],
    embeddings: dict[str, list[float]],
    sim_threshold: float = 0.28,
) -> list[str]:
    """用 embedding 相似度过滤同一课程里的跨学科异常概念。

    对于每个候选 X，计算它与课程内其他候选（不含目标）的平均 embedding 相似度。
    如果平均相似度低于 sim_threshold，认为它是被错误标注进该课程的跨学科概念，
    予以剔除。
    """
    if not candidates or target not in embeddings:
        return candidates

    others = [c for c in candidates if c != target and c in embeddings]
    if len(others) <= 2:
        return candidates

    other_vecs = [np.array(embeddings[c], dtype=np.float32) for c in others]

    def _avg_sim(name: str) -> float:
        if name not in embeddings:
            return 0.0
        vec = np.array(embeddings[name], dtype=np.float32)
        sims = []
        for other, other_vec in zip(others, other_vecs):
            if other == name:
                continue
            sim = _cosine_sim(vec, other_vec)
            if sim is not None:
                sims.append(sim)
        return sum(sims) / len(sims) if sims else 0.0

    # 计算整体平均相似度作为参考
    all_sims = []
    for i, vi in enumerate(other_vecs):
        for j, vj in enumerate(other_vecs):
            if i >= j:
                continue
            sim = _cosine_sim(vi, vj)
            if sim is not None:
                all_sims.append(sim)
    overall_avg = sum(all_sims) / len(all_sims) if all_sims else 0.0

    kept = []
    for c in candidates:
        if c == target:
            continue
        avg = _avg_sim(c)
        # 保留：明显高于阈值，或不显著低于整体平均
        if avg >= sim_threshold or avg >= overall_avg * 0.6:
            kept.append(c)
    return kept


def _get_course_based_candidates(
    session,
    target: str,
    embeddings: dict[str, list[float]],
    exclude: set[str],
    top_k: int = DEFAULT_TOP_K,
    max_total: int = 30,
    max_course_size: int = 30,
    max_concept_coverage: int = 25,
) -> list[str]:
    """基于课程结构为 target 生成前置知识候选，不使用 embedding 相似度。

    候选来源（严格限制，只从真实课程关系里取）：
    1. 覆盖 target 的课程的显式前置课程（PREREQUISITE_FOR）所覆盖的知识点。
    2. 与 target 被同一门课程覆盖的知识点（同课共现）。
       会跳过覆盖知识点数量超过 max_course_size 的课程（数据标注错误）。

    额外过滤：
    - 剔除被超过 max_concept_coverage 门课程覆盖的通用/错误标注概念。
    - 用 embedding 异常检测剔除同一课程内的跨学科噪音（仅用于过滤，不作为候选源）。

    注意：
    - 不再把 embedding 相似度作为候选源。
    - 不再使用「同专业所有课程」，因为专业里可能包含商学、通识等无关课程，
      会污染候选池。
    """
    candidates: list[str] = []
    seen = set(exclude)

    def _add(names: list[str]):
        for name in names:
            if name and name not in seen:
                seen.add(name)
                candidates.append(name)

    # 1. 显式前置课程覆盖的知识点（质量最高）
    result = session.run(
        """
        MATCH (k:KnowledgeConcept {name: $target})<-[:COVERS_KNOWLEDGE]-(c:Course)
        WITH c, count { MATCH (c)-[:COVERS_KNOWLEDGE]->(:KnowledgeConcept) } AS course_size
        WHERE course_size <= $max_course_size
        OPTIONAL MATCH (pc:Course)-[:PREREQUISITE_FOR]->(c)
        WITH pc, count { MATCH (pc)-[:COVERS_KNOWLEDGE]->(:KnowledgeConcept) } AS pc_size
        WHERE pc_size <= $max_course_size
        OPTIONAL MATCH (pc)-[:COVERS_KNOWLEDGE]->(pre:KnowledgeConcept)
        RETURN pre.name AS name
        """,
        target=target,
        max_course_size=max_course_size,
    )
    _add([r["name"] for r in result if r["name"]])

    # 2. 同课共现知识点
    result = session.run(
        """
        MATCH (k:KnowledgeConcept {name: $target})<-[:COVERS_KNOWLEDGE]-(c:Course)
        WITH c, count { MATCH (c)-[:COVERS_KNOWLEDGE]->(:KnowledgeConcept) } AS course_size
        WHERE course_size <= $max_course_size
        MATCH (c)-[:COVERS_KNOWLEDGE]->(co:KnowledgeConcept)
        WHERE co.name <> $target
        RETURN co.name AS name
        """,
        target=target,
        max_course_size=max_course_size,
    )
    _add([r["name"] for r in result if r["name"]])

    # 3. 剔除被过多课程覆盖的通用/错误标注概念
    if candidates:
        coverage_result = session.run(
            """
            MATCH (k:KnowledgeConcept)<-[:COVERS_KNOWLEDGE]-(c:Course)
            WHERE k.name IN $names
            RETURN k.name AS name, count(DISTINCT c) AS coverage
            """,
            names=list(candidates),
        )
        coverage_map = {row["name"]: row["coverage"] for row in coverage_result}
        candidates = [
            c for c in candidates
            if coverage_map.get(c, 0) <= max_concept_coverage
        ]

    # 4. 过滤同一课程内的跨学科异常概念（如 ML 课程里混进「工作与不平等」）
    candidates = _filter_outliers(target, candidates, embeddings)

    return candidates[:max_total]


LLM_PREREQ_BATCH_PROMPT = """你是一位高等教育课程专家。请为下面每个目标知识点，从对应的候选列表中选出明确的前置知识（即学习该目标之前通常需要先学的内容）。

{items}

要求：
1. 只选择确实具有前置依赖关系的项目，不要选择仅仅是同领域、相似词或同一上级概念下的项目。
2. 如果某个候选是目标知识点的子概念、特例、应用场景或后续扩展，则不要选。
3. 必须严格区分不同学科：
   - 人工智能/机器学习中的「神经网络」「深度学习」「卷积神经网络」等，前置应该是数学（线性代数、概率论、优化）、编程、机器学习基础等，而不是生物学的「神经元」「神经科学」。
   - 生物/心理/医学中的「神经元」「神经科学」等，前置才是生物学、心理学、解剖学等。
4. 方向必须正确：如果 A 是 B 的前置，则输出 "A"，不要反向把 B 作为 A 的前置。
5. 宁缺毋滥：不确定时返回空列表，不要把相似概念误当前置。

返回 JSON 格式：
{{
  "目标知识点1": {{
    "prerequisites": ["候选名1", "候选名2"],
    "confidence": "high|medium|low",
    "reasoning": "简要说明"
  }},
  ...
}}

只输出 JSON，不要 Markdown 代码块。"""


def predict_prereqs_batch_llm(
    targets_with_candidates: list[tuple[str, list[str]]],
) -> dict[str, list[tuple[str, float]]]:
    """一次 LLM 调用处理多个目标知识点。

    返回 {target: [(prereq_name, score), ...]}。
    """
    if not os.getenv("DEEPSEEK_API_KEY") or not targets_with_candidates:
        return {}

    items_text = []
    for target, candidates in targets_with_candidates:
        items_text.append(f"目标：{target}\n候选：\n" + "\n".join(f"- {c}" for c in candidates))

    prompt = LLM_PREREQ_BATCH_PROMPT.format(items="\n\n".join(items_text))

    try:
        client = get_deepseek_client()
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": "你是一位严谨的教育知识图谱专家。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            timeout=60,
        )
        raw = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[LLMPrereq] DeepSeek batch call failed: {e}")
        return {}

    parsed = _parse_llm_json(raw)
    if not isinstance(parsed, dict):
        return {}

    results: dict[str, list[tuple[str, float]]] = {}
    valid_targets = {t for t, _ in targets_with_candidates}
    candidate_map = {t: set(cs) for t, cs in targets_with_candidates}

    for target, item in parsed.items():
        if target not in valid_targets:
            continue
        if not isinstance(item, dict):
            continue
        base_score = _confidence_to_score(item.get("confidence", "medium"))
        confirmed = item.get("prerequisites", [])
        prereqs = []
        for name in confirmed:
            name = name.strip()
            if name in candidate_map.get(target, set()):
                prereqs.append((name, base_score))
        results[target] = prereqs

    return results


def predict_prereqs_for_target_llm(
    target: str,
    candidate_names: list[str],
) -> list[tuple[str, float]]:
    """调用 DeepSeek，返回 LLM 确认的前置知识点及其置信度分数。

    如果未配置 API key 或调用失败，返回空列表。
    """
    if not os.getenv("DEEPSEEK_API_KEY") or not candidate_names:
        return []

    candidates_text = "\n".join(f"- {name}" for name in candidate_names)
    prompt = LLM_PREREQ_PROMPT.format(target=target, candidates=candidates_text)

    try:
        client = get_deepseek_client()
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": "你是一位严谨的教育知识图谱专家。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            timeout=30,
        )
        raw = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[LLMPrereq] DeepSeek call failed for {target}: {e}")
        return []

    parsed = _parse_llm_json(raw)
    if not parsed:
        return []

    base_score = _confidence_to_score(parsed.get("confidence", "medium"))
    confirmed = parsed.get("prerequisites", [])

    results = []
    for name in confirmed:
        name = name.strip()
        if name in candidate_names:
            results.append((name, base_score))
    return results


def complete_prerequisites_with_llm(
    targets: Optional[list[str]] = None,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = 0.6,
    max_targets: Optional[int] = None,
    dry_run: bool = False,
) -> dict:
    """批量用 LLM 补全前置关系。

    候选生成策略（优先课程结构，embedding 仅兜底）：
    1. 覆盖目标知识点的课程的显式前置课程所覆盖的知识点；
    2. 与目标知识点同课共现的知识点；
    3. 目标知识点所在专业的其他课程所覆盖的知识点；
    4. 数量不足时再用 embedding 相似度补充。

    这样避免只靠 embedding 把名字相似但语义不同的概念（如「神经网络」
    与「神经科学」）混在一起。

    参数：
        targets: 指定要补全的目标知识点列表；None 表示处理全部知识点。
        top_k: 每个目标知识点至少保留多少个候选。
        min_score: 低于该分数的预测不写入 Neo4j。
        max_targets: 最多处理多少个目标（用于控制 API 调用量和成本）。
        dry_run: True 时只返回预览，不写入数据库。

    返回：
        dict，包含 status、predicted_count、targets_processed、sample_predictions。
    """
    embedder = KnowledgeEmbedder()
    embeddings = embedder.compute_embeddings()
    if not embeddings:
        return {
            "status": "error",
            "message": "Embeddings unavailable.",
            "predicted_count": 0,
            "targets_processed": 0,
        }

    driver = get_neo4j_driver()
    with driver.session() as session:
        if targets:
            all_targets = targets
        else:
            result = session.run(f"MATCH (k:{LABEL_KNOWLEDGE_POINT}) RETURN k.name AS name")
            all_targets = [r["name"] for r in result]

    if max_targets:
        all_targets = all_targets[:max_targets]

    # 读取已有边，避免重复预测
    with driver.session() as session:
        result = session.run(
            f"""
            MATCH (a:{LABEL_KNOWLEDGE_POINT})-[r:{REL_PREREQUISITE_OF}|{REL_PREDICTED_PREREQ}]->(b:{LABEL_KNOWLEDGE_POINT})
            RETURN a.name AS src, b.name AS dst
            """
        )
        existing_edges = set((r["src"], r["dst"]) for r in result)

    predictions = []
    batch: list[tuple[str, list[str]]] = []
    for target in all_targets:
        exclude = {src for src, dst in existing_edges if dst == target} | {target}
        with driver.session() as session:
            candidates = _get_course_based_candidates(
                session, target, embeddings, exclude, top_k=top_k
            )
        if not candidates:
            continue
        batch.append((target, candidates))

        if len(batch) >= DEFAULT_BATCH_SIZE:
            batch_results = predict_prereqs_batch_llm(batch)
            for tgt, prereqs in batch_results.items():
                for src, score in prereqs:
                    if (src, tgt) in existing_edges:
                        continue
                    if score < min_score:
                        continue
                    if _is_bogus_edge(src, tgt):
                        continue
                    predictions.append((src, tgt, score))
            batch = []

    # Process remaining batch
    if batch:
        batch_results = predict_prereqs_batch_llm(batch)
        for tgt, prereqs in batch_results.items():
            for src, score in prereqs:
                if (src, tgt) in existing_edges:
                    continue
                if score < min_score:
                    continue
                if _is_bogus_edge(src, tgt):
                    continue
                predictions.append((src, tgt, score))

    # 去重并限制数量
    predictions = list(dict.fromkeys(predictions))

    stored_count = 0
    if not dry_run and predictions:
        with driver.session() as session:
            for src, dst, score in predictions:
                session.run(
                    f"""
                    MATCH (a:{LABEL_KNOWLEDGE_POINT} {{name: $src}})
                    MATCH (b:{LABEL_KNOWLEDGE_POINT} {{name: $dst}})
                    MERGE (a)-[r:{REL_PREDICTED_PREREQ}]->(b)
                    ON CREATE SET r.confidence = $score,
                                  r.method = $method,
                                  r.created_at = datetime()
                    ON MATCH SET  r.confidence = $score,
                                  r.method = $method,
                                  r.updated_at = datetime()
                    """,
                    src=src,
                    dst=dst,
                    score=round(score, 6),
                    method="deepseek_llm",
                )
                stored_count += 1

    return {
        "status": "dry_run" if dry_run else "success",
        "method": "deepseek_llm",
        "targets_processed": len(all_targets),
        "predicted_count": len(predictions),
        "stored_count": stored_count,
        "sample_predictions": [
            {"src": s, "dst": d, "score": round(sc, 4)}
            for s, d, sc in predictions[:10]
        ],
    }


def infer_real_prerequisites_with_llm(
    targets: Optional[list[str]] = None,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = 0.75,
    max_targets: Optional[int] = None,
    dry_run: bool = False,
) -> dict:
    """用 LLM 从课程结构中推断真实前置关系，并写入 CONCEPT_PREREQUISITE_FOR。

    与 complete_prerequisites_with_llm 不同：这里把 LLM 确认的前置边直接作为
    真实边写入（REL_PREREQUISITE_OF / CONCEPT_PREREQUISITE_FOR），而不是
    PREDICTED_PREREQUISITE。因此 min_score 默认更高，要求更严格。

    参数：
        targets: 指定目标知识点列表；None 表示处理全部。
        top_k: 每个目标保留多少候选。
        min_score: 写入真实边的最低置信度（默认 0.75，比预测边更严格）。
        max_targets: 最多处理多少个目标。
        dry_run: True 时只返回预览，不写入数据库。

    返回：
        dict，包含 status、inferred_count、targets_processed、sample_inferred。
    """
    embedder = KnowledgeEmbedder()
    embeddings = embedder.compute_embeddings()
    if not embeddings:
        return {
            "status": "error",
            "message": "Embeddings unavailable.",
            "inferred_count": 0,
            "targets_processed": 0,
        }

    driver = get_neo4j_driver()
    with driver.session() as session:
        if targets:
            all_targets = targets
        else:
            result = session.run(f"MATCH (k:{LABEL_KNOWLEDGE_POINT}) RETURN k.name AS name")
            all_targets = [r["name"] for r in result]

    if max_targets:
        all_targets = all_targets[:max_targets]

    # 读取已有真实边，避免重复
    with driver.session() as session:
        result = session.run(
            f"""
            MATCH (a:{LABEL_KNOWLEDGE_POINT})-[r:{REL_PREREQUISITE_OF}]->(b:{LABEL_KNOWLEDGE_POINT})
            RETURN a.name AS src, b.name AS dst
            """
        )
        existing_edges = set((r["src"], r["dst"]) for r in result)

    inferred: list[tuple[str, str, float]] = []
    batch: list[tuple[str, list[str]]] = []

    for target in all_targets:
        exclude = {src for src, dst in existing_edges if dst == target} | {target}
        with driver.session() as session:
            candidates = _get_course_based_candidates(
                session, target, embeddings, exclude, top_k=top_k
            )
        if not candidates:
            continue
        batch.append((target, candidates))

        if len(batch) >= DEFAULT_BATCH_SIZE:
            batch_results = predict_prereqs_batch_llm(batch)
            for tgt, prereqs in batch_results.items():
                for src, score in prereqs:
                    if (src, tgt) in existing_edges:
                        continue
                    if score < min_score:
                        continue
                    if _is_bogus_edge(src, tgt):
                        continue
                    inferred.append((src, tgt, score))
            batch = []

    if batch:
        batch_results = predict_prereqs_batch_llm(batch)
        for tgt, prereqs in batch_results.items():
            for src, score in prereqs:
                if (src, tgt) in existing_edges:
                    continue
                if score < min_score:
                    continue
                if _is_bogus_edge(src, tgt):
                    continue
                inferred.append((src, tgt, score))

    inferred = list(dict.fromkeys(inferred))

    stored_count = 0
    if not dry_run and inferred:
        with driver.session() as session:
            for src, dst, score in inferred:
                session.run(
                    f"""
                    MATCH (a:{LABEL_KNOWLEDGE_POINT} {{name: $src}})
                    MATCH (b:{LABEL_KNOWLEDGE_POINT} {{name: $dst}})
                    MERGE (a)-[r:{REL_PREREQUISITE_OF}]->(b)
                    ON CREATE SET r.source = $source,
                                  r.confidence = $score,
                                  r.created_at = datetime()
                    ON MATCH SET  r.source = $source,
                                  r.confidence = $score,
                                  r.updated_at = datetime()
                    """,
                    src=src,
                    dst=dst,
                    score=round(score, 6),
                    source="llm_course_structure_inference",
                )
                stored_count += 1

    return {
        "status": "dry_run" if dry_run else "success",
        "method": "deepseek_llm_course_structure",
        "targets_processed": len(all_targets),
        "inferred_count": len(inferred),
        "stored_count": stored_count,
        "sample_inferred": [
            {"src": s, "dst": d, "score": round(sc, 4)}
            for s, d, sc in inferred[:10]
        ],
    }
