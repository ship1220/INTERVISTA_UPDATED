# skill_profile_integration.py
# Bridge layer: maps interview evaluation results → UserSkillProfile updates → DB persistence
# Drop this file into your project and call update_profile_after_interview() after every interview.

from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

from user_skill_profile import (
    UserSkillProfile,
    UserSkillVector,
    InterviewRecord,
    ScoreBreakdown,
    TechnicalSkillVector,
    InterviewSkillVector,
    CommunicationSkillVector,
    update_skill_score,
    calculate_overall_score,
    detect_weaknesses,
    recommend_micro_courses,
)


# ============================================================
# TOPIC → SKILL FIELD MAPPING
# ============================================================
# Maps weak_topics strings returned by evaluate_content()
# to the exact field names on TechnicalSkillVector /
# InterviewSkillVector / CommunicationSkillVector.

TOPIC_TO_SKILL_FIELD: dict[str, str] = {
    # Technical
    "dsa": "dsa",
    "data structures": "dsa",
    "algorithms": "dsa",
    "dbms": "dbms",
    "database": "dbms",
    "sql": "dbms",
    "operating systems": "operating_systems",
    "os": "operating_systems",
    "processes": "operating_systems",
    "threads": "operating_systems",
    "memory management": "operating_systems",
    "computer networks": "computer_networks",
    "networking": "computer_networks",
    "tcp": "computer_networks",
    "http": "computer_networks",
    "system design": "system_design",
    "scalability": "system_design",
    "architecture": "system_design",
    # Interview
    "relevance": "relevance",
    "explanation": "explanation_depth",
    "explanation depth": "explanation_depth",
    "structured thinking": "structured_thinking",
    "problem solving": "problem_solving",
    "star method": "star_method",
    "behavioral": "star_method",
    # Communication
    "clarity": "clarity",
    "confidence": "confidence",
    "engagement": "engagement",
    "speaking pace": "speaking_pace",
    "pace": "speaking_pace",
    "filler words": "filler_control",
    "filler control": "filler_control",
}


def _resolve_field(topic: str) -> Optional[str]:
    """Return the model field name for a given topic string, or None."""
    return TOPIC_TO_SKILL_FIELD.get(topic.lower().strip())


# ============================================================
# DERIVE SKILL DELTAS FROM EVALUATION
# ============================================================

def _derive_skill_updates(
    evaluation: dict,
    speech_analyses: list[dict],
    confidence_score: int,
) -> dict[str, float]:
    """
    Convert raw evaluation output + speech stats into
    {field_name: new_score} updates for the skill vector.

    evaluation keys (from evaluate_content / interview_evaluation_prompt):
        answers[].score, answers[].weak_topics, weak_topics
    speech_analyses keys (from analyze_speech_delivery):
        clarity_score, engagement_score, speaking_pace_wpm,
        filler_word_count, word_count
    """
    updates: dict[str, float] = {}

    answers: list[dict] = evaluation.get("answers", [])

    # ── Technical / Interview scores from per-question scores ──────────────
    # Map each answered question's score back to a skill field via weak_topics.
    # Questions that identify NO weak topic contribute to a general pool.
    general_scores: list[float] = []

    for ans in answers:
        score: float = float(ans.get("score", 50))
        weak: list[str] = ans.get("weak_topics", [])

        mapped = False
        for topic in weak:
            field = _resolve_field(topic)
            if field:
                # Accumulate — average later if multiple answers hit same field
                updates.setdefault(field, []).append(score)  # type: ignore[assignment]
                mapped = True

        if not mapped:
            general_scores.append(score)

    # Average accumulated lists
    for field, scores in list(updates.items()):
        if isinstance(scores, list):
            updates[field] = round(sum(scores) / len(scores), 1)

    # Overall weak_topics from the evaluation also carry signal
    for topic in evaluation.get("weak_topics", []):
        field = _resolve_field(topic)
        if field and field not in updates:
            # Penalise: identified as globally weak → pull score down slightly
            updates[field] = max(0, updates.get(field, 50) - 10)

    # ── Communication scores from speech analysis ──────────────────────────
    if speech_analyses:
        avg_clarity = sum(a["clarity_score"] for a in speech_analyses) / len(speech_analyses)
        avg_engagement = sum(a["engagement_score"] for a in speech_analyses) / len(speech_analyses)

        total_fillers = sum(a["filler_word_count"] for a in speech_analyses)
        total_words = sum(a["word_count"] for a in speech_analyses)
        filler_rate = total_fillers / max(total_words, 1)
        filler_control = max(0, min(100, round(100 - filler_rate * 300)))

        avg_wpm = sum(a["speaking_pace_wpm"] for a in speech_analyses) / len(speech_analyses)
        # Ideal pace 120–160 wpm → 90 pts; outside → scaled penalty
        if 120 <= avg_wpm <= 160:
            pace_score = 90
        else:
            deviation = min(abs(avg_wpm - 120), abs(avg_wpm - 160))
            pace_score = max(0, round(90 - deviation * 0.5))

        updates["clarity"] = round(avg_clarity, 1)
        updates["engagement"] = round(avg_engagement, 1)
        updates["filler_control"] = filler_control
        updates["speaking_pace"] = pace_score
        updates["confidence"] = confidence_score

    return updates


# ============================================================
# APPLY UPDATES TO SKILL VECTOR (weighted rolling average)
# ============================================================

def _apply_updates_to_vector(
    profile: UserSkillProfile,
    updates: dict[str, float],
) -> UserSkillProfile:
    """
    For each field in updates, blend the new score with the existing
    score using update_skill_score() (70% old / 30% new) so a single
    bad interview doesn't crater a previously high score.
    """
    tech = profile.technical_skills
    inter = profile.interview_skills
    comm = profile.communication_skills

    for field, new_score in updates.items():
        if hasattr(tech, field):
            current = getattr(tech, field)
            setattr(tech, field, update_skill_score(current, new_score))

        elif hasattr(inter, field):
            current = getattr(inter, field)
            setattr(inter, field, update_skill_score(current, new_score))

        elif hasattr(comm, field):
            current = getattr(comm, field)
            setattr(comm, field, update_skill_score(current, new_score))

    profile.technical_skills = tech
    profile.interview_skills = inter
    profile.communication_skills = comm

    return profile


# ============================================================
# BUILD InterviewRecord
# ============================================================

def _build_interview_record(
    question: str,
    topic: str,
    answer_transcript: str,
    score: float,
    strengths: list[str],
    weaknesses: list[str],
    clarity: float,
) -> InterviewRecord:
    breakdown = ScoreBreakdown(
        correctness=score,
        conceptual_depth=min(100, score * 1.05),   # slight boost for depth
        clarity=clarity,
        feedback="; ".join(weaknesses) if weaknesses else "Good answer.",
    )
    return InterviewRecord(
        question=question,
        topic=topic,
        answer_transcript=answer_transcript,
        evaluation_score=score,
        score_breakdown=breakdown,
    )


# ============================================================
# MAIN PUBLIC FUNCTION
# ============================================================

def update_profile_after_interview(
    profile: UserSkillProfile,
    evaluation: dict,
    questions_answers: list[dict],
    speech_analyses: list[dict],
    confidence_score: int,
) -> tuple[UserSkillProfile, list[str], list[dict]]:
    """
    Call this after every completed interview.

    Parameters
    ----------
    profile           : existing UserSkillProfile (loaded from DB)
    evaluation        : result of evaluate_content() or evaluate_answers()
    questions_answers : list of {"question": ..., "answer": ...}
    speech_analyses   : list of analyze_speech_delivery() results (one per answer)
    confidence_score  : result of compute_confidence_score()

    Returns
    -------
    updated_profile   : UserSkillProfile with blended scores + new InterviewRecord
    weaknesses        : list of field names that are still below 40
    recommendations   : list of {"topic": ..., "course": ...} micro-course dicts
    """

    # 1. Derive score deltas
    updates = _derive_skill_updates(evaluation, speech_analyses, confidence_score)

    # 2. Blend into profile skill vector
    profile = _apply_updates_to_vector(profile, updates)

    # 3. Recompute overall score from the single source of truth
    skill_vector = UserSkillVector(
        technical_skills=profile.technical_skills,
        interview_skills=profile.interview_skills,
        communication_skills=profile.communication_skills,
    )
    profile.overall_score = calculate_overall_score(skill_vector)

    # 4. Append InterviewRecord for each Q&A pair
    answers = evaluation.get("answers", [])
    avg_clarity = (
        sum(a["clarity_score"] for a in speech_analyses) / len(speech_analyses)
        if speech_analyses else 50.0
    )

    for i, qa in enumerate(questions_answers):
        ans_eval = answers[i] if i < len(answers) else {}
        score = float(ans_eval.get("score", 50))
        weak_topics = ans_eval.get("weak_topics", [])
        topic = weak_topics[0] if weak_topics else "general"

        record = _build_interview_record(
            question=qa.get("question", ""),
            topic=topic,
            answer_transcript=qa.get("answer", ""),
            score=score,
            strengths=ans_eval.get("strengths", []),
            weaknesses=ans_eval.get("weaknesses", []),
            clarity=avg_clarity,
        )
        profile.interview_history.append(record)

    profile.interview_count += 1
    profile.last_updated = datetime.utcnow()

    # 5. Detect weaknesses + recommend courses
    weaknesses = detect_weaknesses(skill_vector)
    recommendations = recommend_micro_courses(weaknesses)

    return profile, weaknesses, recommendations


# ============================================================
# SERIALISE PROFILE → TEMPLATE-READY DICT
# ============================================================

def profile_to_template_context(profile: UserSkillProfile) -> dict:
    """
    Returns the exact dict that profile.html expects:

        skill_profile          → for radar + breakdown + overall score
        timeline_by_role       → for the performance line chart
        interview_history_by_role → for the history accordion
        improvement_message    → string shown in the improvement card
        top_skill_gaps         → list of {name, score} for the gap section
    """

    # skill_profile
    skill_profile = {
        "overall_score": round(profile.overall_score, 1),
        "technical_skills": profile.technical_skills.model_dump(),
        "interview_skills": profile.interview_skills.model_dump(),
        "communication_skills": profile.communication_skills.model_dump(),
    }

    # timeline_by_role — group InterviewRecord by topic (used as proxy for role)
    # Real role grouping requires storing role on each InterviewRecord;
    # until models.py exposes that, we group by the role in basic_info.
    role = profile.basic_info.target_role or "General"
    timeline_points = []
    for idx, record in enumerate(profile.interview_history, 1):
        timeline_points.append({
            "label": f"Interview {idx}",
            "score": round(record.evaluation_score, 1),
        })
    timeline_by_role = {role: timeline_points} if timeline_points else {}

    # interview_history_by_role — for the accordion in the HTML
    # HTML uses iv.id, iv.date, iv.score — attach lightweight dicts
    history_by_role: dict[str, list] = {}
    for idx, record in enumerate(profile.interview_history):
        history_by_role.setdefault(role, []).append({
            "id": idx + 1,
            "date": record.score_breakdown.timestamp,
            "score": record.evaluation_score,
        })

    # top_skill_gaps — all fields below 60, sorted ascending
    all_skills: dict[str, float] = {
        **profile.technical_skills.model_dump(),
        **profile.interview_skills.model_dump(),
        **profile.communication_skills.model_dump(),
    }
    gaps = sorted(
        [{"name": k.replace("_", " ").title(), "score": v}
         for k, v in all_skills.items() if v < 60],
        key=lambda x: x["score"],
    )

    # improvement_message
    if gaps:
        weakest = gaps[0]["name"]
        improvement_message = (
            f"Your weakest area is {weakest} ({gaps[0]['score']}%). "
            f"Focus on strengthening it before your next interview."
        )
    else:
        improvement_message = (
            "Great job! All your skills are above 60%. "
            "Keep practising to push them closer to 100%."
        )

    return {
        "skill_profile": skill_profile,
        "timeline_by_role": timeline_by_role,
        "interview_history_by_role": history_by_role,
        "top_skill_gaps": gaps,
        "improvement_message": improvement_message,
    }


# ============================================================
# DB PERSISTENCE HELPERS (SQLAlchemy-agnostic)
# ============================================================

def save_profile_to_db(db_session, db_model_instance, profile: UserSkillProfile):
    """
    Generic saver. Assumes your SQLAlchemy User model has a
    `skill_profile_json` TEXT column.

    Usage:
        user = db.query(User).filter_by(id=user_id).first()
        save_profile_to_db(db, user, updated_profile)
    """
    db_model_instance.skill_profile_json = profile.model_dump_json()
    db_session.add(db_model_instance)
    db_session.commit()


def load_profile_from_db(db_model_instance) -> Optional[UserSkillProfile]:
    """
    Load a UserSkillProfile from a DB row's skill_profile_json column.
    Returns None if the column is empty / unparsable.
    """
    raw = getattr(db_model_instance, "skill_profile_json", None)
    if not raw:
        return None
    try:
        return UserSkillProfile.model_validate_json(raw)
    except Exception as exc:
        print(f"[skill_profile] Failed to deserialise profile: {exc}")
        return None