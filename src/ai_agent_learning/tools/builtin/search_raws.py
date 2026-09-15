from typing import Any

from ai_agent_learning.memory.embedding import GlmEmbeddingClient
from ai_agent_learning.memory.schema import MemorySearchResult
from ai_agent_learning.memory.store import QdrantMemoryStore
from ai_agent_learning.tools.contracts import RiskLevel, ToolSpec

SPEC = ToolSpec(
    name="search_legal_knowledge",

    description=(
        "从中国法律法规与规章知识库中检索与用户问题相关的法律条文。"
        "本工具只负责检索，不负责判断案件是否合法、责任归属或形成最终法律结论。"
        "调用时可提供 1~5 个不同角度的检索项，每个检索项应保持中性，"
        "用于描述事实、法律关系、法律概念、适用规则或例外条件。"
        "不得为了支持预先形成的结论而构造检索内容。"
        "如果不能确定具体法规名称，title_hint 应留空，不得编造。"
    ),

    risk_level=RiskLevel.LOW,

    input_schema={
        "type": "object",
        "properties": {
            "queries": {
                "type": "array",
                "description": (
                    "1~5 个相互补充的法律检索项。"
                    "复杂问题应尽量从事实、法律关系、法律概念、规则、例外等不同角度检索，"
                    "避免只是对同一句问题进行重复改写。"
                ),
                "minItems": 1,
                "maxItems": 5,
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": [
                                "fact",
                                "relationship",
                                "concept",
                                "rule",
                                "exception",
                            ],
                            "description": (
                                "检索项类型："
                                "fact=客观事实；"
                                "relationship=法律关系；"
                                "concept=中性法律概念；"
                                "rule=需要查找的适用规则或构成条件；"
                                "exception=例外、限制、免责或相反适用条件。"
                            ),
                        },

                        "title_hint": {
                            "type": "string",
                            "description": (
                                "可能相关的法律、法规、司法解释或规章名称。"
                                "仅在较有把握时填写；不确定时传空字符串。"
                                "该字段仅作为标题检索和排序提示，不应作为强制过滤条件。"
                            ),
                        },

                        "query": {
                            "type": "string",
                            "minLength": 1,
                            "description": (
                                "用于正文检索的中性检索文本。"
                                "应尽量包含有区分度的主体、行为、对象、法律关系或法律概念。"
                                "不得加入尚未由用户事实或法律资料支持的结论，"
                                "例如不得擅自写入“违法”“无需赔偿”“合同无效”等结论性表述。"
                            ),
                        },
                    },

                    "required": [
                        "type",
                        "title_hint",
                        "query",
                    ],

                    "additionalProperties": False,
                },
            }
        },

        "required": ["queries"],
        "additionalProperties": False,
    },

    has_side_effects=False,

    # 纯检索工具，重复执行不会改变任何外部状态，
    # 因此可以安全重试。
    supports_idempotent_retry=True,

    # 普通数据库/搜索请求由执行器统一控制超时即可。
    manages_own_timeout=False,
)

