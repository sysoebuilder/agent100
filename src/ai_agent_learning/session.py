from ai_agent_learning.schema import Session, Message, ModelRequest, ModelResponse
import json
import re
from pathlib import Path

SESSION_DIR = Path(__file__).resolve().parents[1] / "data" / "sessions"
def save_session(session: Session) -> Path:
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    file_path = SESSION_DIR / f"{session.session_id}.json"

    with file_path.open("w", encoding="utf-8") as file:
        json.dump(
            session.model_dump(mode="json"),
            file,
            ensure_ascii=False,
            indent=2,
        )

    return file_path

def load_session() -> list[Session]:
    sessions: list[Session] = []

    session_files = sorted(
        SESSION_DIR.glob("*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    for file_path in session_files:
        with file_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        session = Session.model_validate(data)
        sessions.append(session)

    return sessions

def clear_session():
    pass

def create_session_name(prompt: str, max_length: int = 20) -> str:
    text = "".join(prompt.split())

    first_sentence = re.split(
            r"[。！？!?；;\n]",
            text,
            maxsplit=1,
        )[0]

    name = first_sentence[:max_length].strip()

    return name

def select_session() -> Session:
    sessions = load_session()

    print("0. 创建新会话")

    for index, session in enumerate(sessions, start=1):
        print(f"{index}. {session.name}")

    while True:
        choice = input("请选择会话编号: ").strip()

        if choice == "0":
            return Session()

        if not choice.isdigit():
            print("请输入数字")
            continue

        index = int(choice) - 1

        if 0 <= index < len(sessions):
            return sessions[index]

        print("会话编号不存在")