# 聊天终端界面

运行 `agent100 chat`。界面使用 Rich，展示会话列表、模型和会话标题、用户输入分隔线、助手正文及错误提示。

GLM 流式响应的 `reasoning_content` 通过 `on_reasoning_delta` 传给临时思考区。只展示最近几行，按中文显示宽度折行，避免长思考内容滚入终端历史。正文开始时替换整个思考区，直接以 Markdown 逐块渲染正文；流式阶段与最终结果共用同一套“助手”标题和排版，没有正文外框。响应结束后保留完整内容，包括代码高亮、列表和表格。最终终端中留下正文，不留下思考区。

UI 的思考缓存只用于显示；Client 仍独立保留完整思考内容用于工具续调。工具名称、参数、结果不展示。有工具但没有正文的响应也会触发 `on_response_end()`，清理当前思考区；工具执行期间只显示通用等待状态。

中断或异常时清除临时显示，已收到的部分正文标注为“回复未完成”。请求失败后显示错误并允许继续输入。界面不改变 taskplan 的输出方式。

重定向输出或不支持光标控制的终端无法可靠擦除思考，因此自动省略临时思考区，正文仍逐块写出。UI 显示的是 API 返回的思考字段，没有返回该字段时只显示等待状态。

实现依据：[Rich Live 临时显示](https://rich.readthedocs.io/en/stable/live.html#transient-display)、[Markdown 渲染](https://rich.readthedocs.io/en/stable/markdown.html)。
