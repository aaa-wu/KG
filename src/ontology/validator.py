"""人机复核队列：LLM 抽取结果需要老师/管理员批准后才能导入 Neo4j。"""
import json
import os
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

PENDING_VALIDATION_FILE = "data/pending_validation.jsonl"
VALIDATED_TOPICS_FILE = "data/validated_topics.jsonl"


@dataclass
class ValidationItem:
    id: str
    major: str
    domain: str
    topic: str
    subtopics: list[str]
    course_mappings: list[str]  # 映射到该 Topic 的课程名
    concept_mappings: list[str]  # 映射到该 Topic 下 SubTopic 的知识点
    status: str  # pending | approved | rejected | modified
    validator_notes: str
    created_at: str
    validated_at: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ValidationItem":
        return cls(**d)


def _ensure_files():
    os.makedirs(os.path.dirname(PENDING_VALIDATION_FILE), exist_ok=True)
    for path in [PENDING_VALIDATION_FILE, VALIDATED_TOPICS_FILE]:
        if not os.path.exists(path):
            open(path, "a", encoding="utf-8").close()


def _read_all_items(path: str) -> list[ValidationItem]:
    if not os.path.exists(path):
        return []
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(ValidationItem.from_dict(json.loads(line)))
            except Exception:
                continue
    return items


def _write_items(path: str, items: list[ValidationItem]):
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")


def queue_extraction_result(result: "LLMExtractionResult") -> list[str]:
    """把一个 Major 的抽取结果拆成多个 ValidationItem 写入待审队列。"""
    from src.ontology.topic_extractor import LLMExtractionResult

    _ensure_files()
    item_ids = []
    existing_items = _read_all_items(PENDING_VALIDATION_FILE)
    existing_ids = {item.id for item in existing_items}

    # 课程 -> topic 反查
    topic_to_courses: dict[str, list[str]] = {}
    for course, topics in result.course_to_topics.items():
        for t in topics:
            topic_to_courses.setdefault(t, []).append(course)

    # 知识点 -> subtopic 反查
    subtopic_to_concepts: dict[str, list[str]] = {}
    for concept, subtopics in result.concept_to_subtopics.items():
        for st in subtopics:
            subtopic_to_concepts.setdefault(st, []).append(concept)

    new_items = []
    for topic in result.topics:
        item_id = f"{result.major}_{topic.name}_{uuid.uuid4().hex[:8]}"
        if item_id in existing_ids:
            continue
        # 该 topic 下的概念：把每个 subtopic 对应的概念收集起来
        concept_mappings = []
        for st in topic.subtopics:
            concept_mappings.extend(subtopic_to_concepts.get(st, []))
        concept_mappings = list(dict.fromkeys(concept_mappings))[:30]

        item = ValidationItem(
            id=item_id,
            major=result.major,
            domain=topic.domain or result.major,
            topic=topic.name,
            subtopics=topic.subtopics,
            course_mappings=topic_to_courses.get(topic.name, []),
            concept_mappings=concept_mappings,
            status="pending",
            validator_notes="",
            created_at=datetime.utcnow().isoformat(),
        )
        new_items.append(item)
        item_ids.append(item_id)

    with open(PENDING_VALIDATION_FILE, "a", encoding="utf-8") as f:
        for item in new_items:
            f.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")

    return item_ids


def get_pending_validations(limit: int = 100) -> list[ValidationItem]:
    _ensure_files()
    items = [item for item in _read_all_items(PENDING_VALIDATION_FILE) if item.status == "pending"]
    return items[:limit]


def approve_extraction(item_id: str, notes: str = "") -> bool:
    _ensure_files()
    pending = _read_all_items(PENDING_VALIDATION_FILE)
    target = None
    for item in pending:
        if item.id == item_id:
            target = item
            break
    if target is None:
        return False

    target.status = "approved"
    target.validator_notes = notes
    target.validated_at = datetime.utcnow().isoformat()

    _write_items(PENDING_VALIDATION_FILE, pending)

    validated = _read_all_items(VALIDATED_TOPICS_FILE)
    validated.append(target)
    _write_items(VALIDATED_TOPICS_FILE, validated)
    return True


def reject_extraction(item_id: str, notes: str = "") -> bool:
    _ensure_files()
    pending = _read_all_items(PENDING_VALIDATION_FILE)
    for item in pending:
        if item.id == item_id:
            item.status = "rejected"
            item.validator_notes = notes
            item.validated_at = datetime.utcnow().isoformat()
            _write_items(PENDING_VALIDATION_FILE, pending)
            return True
    return False


def modify_extraction(item_id: str, topic: str, subtopics: list[str], notes: str = "") -> bool:
    _ensure_files()
    pending = _read_all_items(PENDING_VALIDATION_FILE)
    for item in pending:
        if item.id == item_id:
            item.topic = topic
            item.subtopics = subtopics
            item.status = "modified"
            item.validator_notes = notes
            item.validated_at = datetime.utcnow().isoformat()
            _write_items(PENDING_VALIDATION_FILE, pending)
            return True
    return False


def load_validated_items(major: Optional[str] = None) -> list[ValidationItem]:
    _ensure_files()
    items = [item for item in _read_all_items(VALIDATED_TOPICS_FILE) if item.status in ("approved", "modified")]
    if major:
        items = [item for item in items if item.major == major]
    return items


def load_all_validated_topics() -> list[ValidationItem]:
    return load_validated_items()
