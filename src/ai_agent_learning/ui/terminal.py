from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from ai_agent_learning.schema import Session


class TerminalUI:
    """思考与生成预览使用可清除区域，完成后只留下 Markdown 正文。"""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()
        self._interactive = (
            self.console.is_terminal and not self.console.is_dumb_terminal
        )
        self._live: Live | None = None
        self._reasoning = ""
        self._text = ""

    def show_header(self, model: str, session_name: str) -> None:
        self.console.print(
            Panel(
                Group(
                    Text.assemble(("agent100", "bold cyan"), "  ·  ", model),
                    Text(f"会话 · {session_name or '新会话'}", style="dim"),
                    Text("输入消息开始对话 · exit / quit 退出", style="dim"),
                ),
                border_style="cyan",
                padding=(0, 2),
            )
        )

    def read_message(self) -> str:
        self.console.print()
        self.console.rule(Text("你", style="bold cyan"), align="left", style="dim")
        return input("› ").strip()

    def show_sessions(self, sessions: list[Session]) -> None:
        table = Table(
            title="选择会话", border_style="dim", show_edge=False, padding=(0, 2)
        )
        table.add_column("编号", style="cyan", justify="right")
        table.add_column("会话")
        table.add_row("0", Text("创建新会话", style="bold"))
        for index, session in enumerate(sessions, start=1):
            table.add_row(str(index), Text(session.name or "未命名会话"))
        self.console.print(table)

    def show_error(self, message: str) -> None:
        self.console.print(Text(f"错误 · {message}", style="bold red"))

    def show_notice(self, message: str) -> None:
        self.console.print(Text(message, style="dim"))

    def begin_turn(self) -> None:
        if self._interactive and self._live is None:
            self._live = Live(
                Spinner("dots", text=Text("正在处理…", style="dim")),
                console=self.console,
                transient=True,
                refresh_per_second=12,
                vertical_overflow="crop",
            )
            self._live.start(refresh=True)

    def _tail(self, text: str, max_lines: int, style: str = "") -> Text:
        # 先按终端的中文字符宽度折行，再限制高度，防止思考内容滚入历史区。
        lines = Text(text[-6000:], style=style).wrap(
            self.console,
            max(1, self.console.width - 4),
            overflow="fold",
        )
        return Text("\n").join(lines[-max_lines:])

    def reasoning_delta(self, text: str) -> None:
        if not text or not self._interactive or self._text:
            return
        self._reasoning = (self._reasoning + text)[-6000:]
        self.begin_turn()
        if self._live is not None:
            self._live.update(
                Panel(
                    Group(
                        Spinner("dots", text=Text("正在思考", style="yellow")),
                        self._tail(
                            self._reasoning,
                            max(1, min(6, self.console.height - 5)),
                            "dim",
                        ),
                    ),
                    border_style="dim",
                    title="思考过程 · 输出正文后自动清除",
                ),
                refresh=True,
            )

    def text_delta(self, text: str) -> None:
        if not text:
            return
        first = not self._text
        self._text += text
        self._reasoning = ""
        if self._interactive:
            self.begin_turn()
            if self._live is not None:
                self._live.update(
                    self._render_answer(),
                    refresh=first,
                )
        else:
            # 重定向输出无法擦除内容：省略思考，正文仍逐块刷新。
            if first:
                self.console.file.write("助手: ")
            self.console.file.write(text)
            self.console.file.flush()

    def _stop_live(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None

    def _render_answer(self, *, interrupted: bool = False) -> Group:
        label = "助手 · 回复未完成" if interrupted else "助手"
        return Group(
            Rule(Text(label, style="bold cyan"), align="left", style="dim"),
            Markdown(self._text, code_theme="monokai"),
            Text(""),
        )

    def _finish_text(self, *, interrupted: bool = False) -> None:
        self._stop_live()
        if self._text:
            if self._interactive:
                self.console.print(self._render_answer(interrupted=interrupted))
            else:
                self.console.file.write("\n")
                self.console.file.flush()
        self._text = ""
        self._reasoning = ""

    def finish_response(self) -> None:
        self._finish_text()
        self.begin_turn()

    def close_turn(self) -> None:
        # 正常路径的正文已提交；异常时保留已收到的正文并标注未完成。
        self._finish_text(interrupted=True)
