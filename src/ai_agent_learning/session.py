import json
import logging
import re
import time
from pathlib import Path

from pydantic import ValidationError

from ai_agent_learning.schema import Session


logger = logging.getLogger(__name__)

SESSION_DIR = Path(__file__).resolve().parents[1] / "data" / "sessions"


def save_session(session: Session) -> Path:
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    file_path = SESSION_DIR / f"{session.session_id}.json"

    try:
        with file_path.open("w", encoding="utf-8") as file:
            json.dump(
                session.model_dump(mode="json"),
                file,
                ensure_ascii=False,
                indent=2,
            )
    except OSError:
        logger.exception(
            "会话保存失败: session_id=%s file_path=%s",
            session.session_id,
            file_path,
            extra={
                "event": "session.save.failed",
                "session_id": session.session_id,
                "file_path": str(file_path),
            },
        )
        raise

    logger.info(
        "会话保存完成: session_id=%s file_path=%s",
        session.session_id,
        file_path,
        extra={
            "event": "session.saved",
            "session_id": session.session_id,
            "file_path": str(file_path),
        },
    )

    return file_path


def load_session() -> list[Session]:
    started_at = time.perf_counter()
    sessions: list[Session] = []

    session_files = sorted(
        SESSION_DIR.glob("*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    for file_path in session_files:
        try:
            with file_path.open("r", encoding="utf-8") as file:
                data = json.load(file)

            session = Session.model_validate(data)
            sessions.append(session)

            logger.debug(
                "会话文件加载成功: session_id=%s file_path=%s",
                session.session_id,
                file_path,
                extra={
                    "event": "session.file.loaded",
                    "session_id": session.session_id,
                    "file_path": str(file_path),
                },
            )
        except (OSError, json.JSONDecodeError, ValidationError) as error:
            logger.warning(
                "会话文件加载失败，已跳过: file_path=%s error_type=%s",
                file_path,
                type(error).__name__,
                extra={
                    "event": "session.file.load_failed",
                    "file_path": str(file_path),
                    "error_type": type(error).__name__,
                },
                exc_info=True,
            )

    duration_ms = round(
        (time.perf_counter() - started_at) * 1000,
        2,
    )

    logger.info(
        "会话加载完成: session_count=%s file_count=%s duration_ms=%s",
        len(sessions),
        len(session_files),
        duration_ms,
        extra={
            "event": "session.load.completed",
            "session_count": len(sessions),
            "file_count": len(session_files),
            "duration_ms": duration_ms,
        },
    )

    return sessions


def clear_session():
    pass


def create_session_name(
    prompt: str,
    max_length: int = 20,
    *,
    session_id: str | None = None,
) -> str:
    if max_length <= 0:
        raise ValueError("max_length 必须大于 0")

    text = "".join(prompt.split())

    first_sentence = re.split(
            r"[。！？!?；;\n]",
            text,
            maxsplit=1,
        )[0]

    name = first_sentence[:max_length].strip()

    logger.info(
        "会话名称创建完成: session_id=%s name_length=%s",
        session_id or "unknown",
        len(name),
        extra={
            "event": "session.name.created",
            "session_id": session_id,
            "name_length": len(name),
        },
    )

    return name


def select_session() -> Session:
    sessions = load_session()

    print("0. 创建新会话")

    for index, session in enumerate(sessions, start=1):
        print(f"{index}. {session.name}")

    while True:
        choice = input("请选择会话编号: ").strip()

        if choice == "0":
            session = Session()
            logger.info(
                "新会话创建完成: session_id=%s",
                session.session_id,
                extra={
                    "event": "session.created",
                    "session_id": session.session_id,
                },
            )
            return session

        if not choice.isdigit():
            print("请输入数字")
            continue

        index = int(choice) - 1

        if 0 <= index < len(sessions):
            session = sessions[index]
            logger.info(
                "会话选择完成: session_id=%s",
                session.session_id,
                extra={
                    "event": "session.selected",
                    "session_id": session.session_id,
                },
            )
            return session

        print("会话编号不存在")
