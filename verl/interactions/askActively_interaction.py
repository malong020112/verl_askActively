import logging
import os
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from verl.interactions.base import BaseInteraction

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AskActivelyInteraction(BaseInteraction):
    """Multi-turn interaction for ask-actively style dialogues.

    Reward per user turn:
    - User replies "ACCEPT": +1 and terminate the dialogue.
    - User replies "REJECT": -1 and continue.
    - Per extra assistant turn (> 1): -0.3 penalty (charged per user turn when assistant_turns > 1).

    If ground_truth is provided via interaction_kwargs, e.g.:
        {"ground_truth": {"user_accept": bool, "optimal_round": int}}
    this interaction will accept at or after the optimal_round when user_accept is True.
    Otherwise, it accepts at the first assistant turn by default (to avoid dead loops).
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._instance_dict: Dict[str, Dict[str, Any]] = {}

        # default reward policy (can be overridden in interaction config or per-sample kwargs)
        rp = (config or {}).get("reward_policy", {})
        self._default_accept_reward: float = float(rp.get("accept_reward", 1.0))
        self._default_rejection_cost: float = float(rp.get("rejection_cost", -1.0))
        self._default_cq_cost: float = float(rp.get("clarifying_question_cost", -0.3))

    async def start_interaction(self, instance_id: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())

        # ground_truth can be passed in interaction_kwargs
        ground_truth = kwargs.get("ground_truth", None)

        # allow overriding reward policy at runtime per sample
        rp = kwargs.get("reward_policy", {}) or {}
        accept_reward = float(rp.get("accept_reward", self._default_accept_reward))
        rejection_cost = float(rp.get("rejection_cost", self._default_rejection_cost))
        cq_cost = float(rp.get("clarifying_question_cost", self._default_cq_cost))

        self._instance_dict[instance_id] = {
            "assistant_turns": 0,
            "num_rejects": 0,
            "last_reward": 0.0,
            "ground_truth": ground_truth,
            "policy": {
                "accept_reward": accept_reward,
                "rejection_cost": rejection_cost,
                "clarifying_question_cost": cq_cost,
            },
        }
        return instance_id

    async def generate_response(
        self, instance_id: str, messages: List[Dict[str, Any]], **kwargs
    ) -> Tuple[bool, str, float, Dict[str, Any]]:
        state = self._instance_dict[instance_id]
        policy = state["policy"]

        # Count assistant turns from current message history
        assistant_turns = sum(1 for m in messages if isinstance(m, dict) and m.get("role") == "assistant")
        state["assistant_turns"] = assistant_turns

        # Determine accept/reject
        accept = False
        gt = state.get("ground_truth") or {}
        if isinstance(gt, dict) and ("user_accept" in gt or "optimal_round" in gt):
            user_accept = bool(gt.get("user_accept", True))
            optimal_round = int(gt.get("optimal_round", 1))
            accept = bool(user_accept and (assistant_turns >= max(1, optimal_round)))
        else:
            # Fallback: accept at first assistant turn to avoid indefinite loops
            accept = assistant_turns >= 1

        # Per-turn penalty if assistant_turns > 1
        turn_penalty = policy["clarifying_question_cost"] if assistant_turns > 1 else 0.0

        if accept:
            reward = policy["accept_reward"] + turn_penalty
            content = "ACCEPT"
            should_terminate = True
        else:
            reward = policy["rejection_cost"] + turn_penalty
            content = "REJECT"
            should_terminate = False
            state["num_rejects"] += 1

        state["last_reward"] = float(reward)

        metrics = {
            "assistant_turns": assistant_turns,
            "num_rejects": state["num_rejects"],
            "turn_penalty": float(turn_penalty),
            "accept": bool(accept),
        }

        return should_terminate, content, float(reward), metrics

    async def calculate_score(self, instance_id: str, **kwargs) -> float:
        # For compatibility: return the last computed reward of this user turn
        return float(self._instance_dict.get(instance_id, {}).get("last_reward", 0.0))

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]

