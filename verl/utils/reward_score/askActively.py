def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    # extra_info 需要包含 messages 和 reward_scores
    extra_info = extra_info or {}
    reward_scores = extra_info.get("reward_scores", {}) or {}
    user_turn_rewards = reward_scores.get("user_turn_rewards", []) or []

    # 统计“助手轮数”= messages 里 role == "assistant" 的条数
    messages = extra_info.get("messages") or []
    raw = extra_info.get("messages")
    if isinstance(raw, list) and raw and isinstance(raw[0], dict) and "messages" in raw[0]: seq = raw[0]["messages"]
    elif isinstance(raw, dict) and "messages" in raw: seq = raw["messages"]
    else: seq = raw or []
    def get_role(m): return getattr(m, "role", None) if not isinstance(m, dict) else m.get("role")
    assistant_turns = sum(1 for m in seq if get_role(m) == "assistant")
    if assistant_turns <= 0: assistant_turns = 1

    # 每多一轮助手回复惩罚 -0.3
    extra_turns = max(assistant_turns - 1, 0)
    penalty = -0.3 * extra_turns

    # 交互器返回的逐轮分数求和（reject=-1、accept=+1）
    user_feedback_sum = float(sum(user_turn_rewards))

    total = user_feedback_sum + penalty
    return {
        "score": float(total),
        "assistant_turns": assistant_turns,
        "extra_turns": extra_turns,
        "penalty": float(penalty),
        "user_feedback_sum": float(user_feedback_sum),
    }
