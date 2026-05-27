"""跨校专业对比：课程重叠率、知识点覆盖差异"""
from src.models.schema import LABEL_MAJOR, LABEL_COURSE, LABEL_KNOWLEDGE_POINT


def compare_majors(session, major_name: str, uni_a: str, uni_b: str) -> dict:
    """对比两所高校同一专业的课程和知识点差异"""

    def _get_major_data(uni: str) -> dict:
        result = session.run(
            f"""
            MATCH (m:{LABEL_MAJOR} {{name: $major, university: $uni}})
            -[:BELONGS_TO]->(c:{LABEL_COURSE})
            OPTIONAL MATCH (c)-[:COVERS]->(k:{LABEL_KNOWLEDGE_POINT})
            RETURN m.name AS major, m.university AS university,
                   collect(DISTINCT c.name) AS courses,
                   collect(DISTINCT k.name) AS knowledge_points
            """,
            major=major_name, uni=uni,
        )
        r = result.single()
        if not r:
            return {"university": uni, "courses": [], "knowledge_points": []}
        return {
            "university": r["university"],
            "courses": r["courses"],
            "knowledge_points": [k for k in r["knowledge_points"] if k],
        }

    data_a = _get_major_data(uni_a)
    data_b = _get_major_data(uni_b)

    courses_a = set(data_a["courses"])
    courses_b = set(data_b["courses"])
    kps_a = set(data_a["knowledge_points"])
    kps_b = set(data_b["knowledge_points"])

    common_courses = courses_a & courses_b
    only_a_courses = courses_a - courses_b
    only_b_courses = courses_b - courses_a
    common_kps = kps_a & kps_b
    only_a_kps = kps_a - kps_b
    only_b_kps = kps_b - kps_a

    def _overlap(a, b):
        total = a | b
        return round(len(a & b) / len(total) * 100, 1) if total else 0

    return {
        "major": major_name,
        "university_a": uni_a,
        "university_b": uni_b,
        "course_count": {"a": len(courses_a), "b": len(courses_b)},
        "kp_count": {"a": len(kps_a), "b": len(kps_b)},
        "course_overlap_rate": f"{_overlap(courses_a, courses_b)}%",
        "kp_overlap_rate": f"{_overlap(kps_a, kps_b)}%",
        "common_courses": sorted(common_courses),
        "common_knowledge_points": sorted(common_kps),
        "only_in_a": {
            "university": uni_a,
            "courses": sorted(only_a_courses),
            "knowledge_points": sorted(only_a_kps),
        },
        "only_in_b": {
            "university": uni_b,
            "courses": sorted(only_b_courses),
            "knowledge_points": sorted(only_b_kps),
        },
    }
