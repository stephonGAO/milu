"""测试 ConversationHistory"""
from milu.llm.base.message import Message, MessageRole
from milu.agent.history import ConversationHistory


def test_add_and_get_messages():
    """应能添加和获取消息"""
    history = ConversationHistory()
    history.set_system(Message(role=MessageRole.SYSTEM, content="你是助手"))
    history.add(Message(role=MessageRole.USER, content="你好"))

    messages = history.get_messages()
    assert len(messages) == 2
    assert messages[0].role == MessageRole.SYSTEM
    assert messages[1].content == "你好"


def test_sliding_window_strategy():
    """滑动窗口应保留 system + 最近 max_turns 条消息"""
    history = ConversationHistory(strategy="sliding_window", max_turns=4)
    history.set_system(Message(role=MessageRole.SYSTEM, content="系统"))

    for i in range(6):
        history.add(Message(role=MessageRole.USER, content=f"消息{i}"))

    messages = history.get_messages()
    assert len(messages) == 5  # system + 最近 4 条
    assert messages[0].role == MessageRole.SYSTEM
    assert messages[1].content == "消息2"
    assert messages[-1].content == "消息5"


def test_none_strategy():
    """none 策略应保留全部消息"""
    history = ConversationHistory(strategy="none")
    history.set_system(Message(role=MessageRole.SYSTEM, content="系统"))

    for i in range(20):
        history.add(Message(role=MessageRole.USER, content=f"消息{i}"))

    messages = history.get_messages()
    assert len(messages) == 21  # system + 20 条


def test_token_limit_strategy():
    """token_limit 应按 token 数截断"""
    history = ConversationHistory(strategy="token_limit", max_tokens=50)
    history.set_system(Message(role=MessageRole.SYSTEM, content="系统"))

    # 添加长消息
    history.add(Message(role=MessageRole.USER, content="这是一条很长的消息" * 10))
    history.add(Message(role=MessageRole.ASSISTANT, content="回复" * 10))
    history.add(Message(role=MessageRole.USER, content="短消息"))

    messages = history.get_messages()
    assert messages[0].role == MessageRole.SYSTEM
    assert messages[-1].content == "短消息"


def test_head_tail_strategy():
    """head_tail 应保留头尾，中间截断"""
    history = ConversationHistory(
        strategy="head_tail",
        head_turns=2,
        tail_turns=2
    )
    history.set_system(Message(role=MessageRole.SYSTEM, content="系统"))

    for i in range(10):
        history.add(Message(role=MessageRole.USER, content=f"消息{i}"))

    messages = history.get_messages()
    # system + head(2) + separator + tail(2)
    assert messages[0].role == MessageRole.SYSTEM
    assert messages[1].content == "消息0"
    assert messages[2].content == "消息1"
    assert "..." in messages[3].content
    assert messages[-2].content == "消息8"
    assert messages[-1].content == "消息9"


def test_clear():
    """clear 应清空历史但保留 system"""
    history = ConversationHistory()
    history.set_system(Message(role=MessageRole.SYSTEM, content="系统"))
    history.add(Message(role=MessageRole.USER, content="你好"))

    history.clear()

    messages = history.get_messages()
    assert len(messages) == 1
    assert messages[0].role == MessageRole.SYSTEM


def test_all_messages_property():
    """all_messages 应返回完整未截断的历史"""
    history = ConversationHistory(strategy="sliding_window", max_turns=2)
    history.set_system(Message(role=MessageRole.SYSTEM, content="系统"))

    for i in range(5):
        history.add(Message(role=MessageRole.USER, content=f"消息{i}"))

    truncated = history.get_messages()
    assert len(truncated) <= 3  # system + max_turns-1

    full = history.all_messages
    assert len(full) == 6  # system + 5
