import logging
import os
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from verl.interactions.base import BaseInteraction

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AskActivelyInteraction(BaseInteraction):
    """Multi-turn user simulator that delegates user decisions to an external LLM API.

    Actions per user turn decided by the external LLM:
    - ACCEPT: +1 and terminate the dialogue.
    - REJECT: -1 and continue.
    - ANSWER (clarification): -0.3 and continue, reply with short answer text.

    The API key, base URL and model are left configurable/blank for you to fill.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._instance_dict: Dict[str, Dict[str, Any]] = {}

        # Reward policy (overridable per-sample via interaction_kwargs.reward_policy)
        rp = (config or {}).get("reward_policy", {})
        self._default_accept_reward: float = float(rp.get("accept_reward", 1.0))
        self._default_rejection_cost: float = float(rp.get("rejection_cost", -1.0))
        self._default_cq_cost: float = float(rp.get("clarifying_question_cost", -0.3))

        # External LLM API config (placeholders, fill in later)
        api_cfg = (config or {}).get("api", {})
        self._api_base_url: str = api_cfg.get("base_url", "")  # e.g., "https://api.openai.com/v1/chat/completions"
        self._api_model: str = api_cfg.get("model", "")        # e.g., "gpt-4o-mini"
        self._api_key_env: str = api_cfg.get("key_env", "USER_LLM_API_KEY")
        self._api_timeout: float = float(api_cfg.get("timeout", 30.0))
        # Optional vendor-specific headers or extra fields
        self._api_extra_headers: Dict[str, str] = api_cfg.get("extra_headers", {})
        self._api_extra_body: Dict[str, Any] = api_cfg.get("extra_body", {})

        # System instruction for the user-simulator (you can refine later)
        self._user_sim_system_prompt: str = (
            (api_cfg.get("system_prompt") or "").strip()
            or "You simulate a human user. Decide to ACCEPT, REJECT, or ANSWER a clarifying question."
            " Return a compact JSON: {\"action\": \"accept|reject|answer\", \"content\": \"<short text>\"}."
            " If accepting, set action=accept and content=\"ACCEPT\"."
            " If rejecting, set action=reject and content=\"REJECT\"."
            " If answering a clarifying question, set action=answer and content to a brief answer (e.g., \"A\")."
        )

    async def start_interaction(self, instance_id: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())

        ground_truth = kwargs.get("ground_truth", None)
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

        assistant_turns = sum(1 for m in messages if isinstance(m, dict) and m.get("role") == "assistant")
        state["assistant_turns"] = assistant_turns

        # Ask external LLM to decide the user action
        try:
            action, content = await self._decide_user_action_with_llm(messages)
        except Exception as e:
            logger.warning(f"User LLM API failed, fallback to REJECT. Error: {e}")
            action, content = "reject", "REJECT"

        action = (action or "").strip().lower()
        if action not in {"accept", "reject", "answer"}:
            # Fallback safety
            action = "reject"
            content = "REJECT"

        # Reward mapping
        if action == "accept":
            reward = policy["accept_reward"]
            should_terminate = True
            content = "ACCEPT"
        elif action == "reject":
            reward = policy["rejection_cost"]
            should_terminate = False
            content = "REJECT"
            state["num_rejects"] += 1
        else:  # answer (clarification)
            reward = policy["clarifying_question_cost"]
            should_terminate = False
            if not content:
                content = "A"  # minimal placeholder answer

        state["last_reward"] = float(reward)

        metrics = {
            "assistant_turns": assistant_turns,
            "num_rejects": state["num_rejects"],
            "action": action,
        }

        return should_terminate, content, float(reward), metrics

    async def calculate_score(self, instance_id: str, **kwargs) -> float:
        return float(self._instance_dict.get(instance_id, {}).get("last_reward", 0.0))

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]

    # ------------------------- internal helpers -------------------------
    async def _decide_user_action_with_llm(self, messages: List[Dict[str, Any]]):
        """Call an external LLM to decide the user action.

        Expects a JSON string response like:
        {"action": "accept|reject|answer", "content": "..."}
        """
        import json
        import os
        import asyncio

        api_key = os.getenv(self._api_key_env, "")
        if not self._api_base_url:
            raise RuntimeError("User LLM base_url is not configured.")
        if not self._api_model:
            raise RuntimeError("User LLM model is not configured.")
        if not api_key:
            raise RuntimeError(f"User LLM API key env '{self._api_key_env}' is not set.")

        user_sim_prompt = self._build_user_sim_prompt(messages)

        payload = {
            "model": self._api_model,
            "messages": [
                {"role": "system", "content": self._user_sim_system_prompt},
                {"role": "user", "content": user_sim_prompt},
            ],
            "temperature": 0.0,
        }
        if self._api_extra_body:
            payload.update(self._api_extra_body)

        # Prefer aiohttp if available, otherwise fallback to urllib in a thread
        try:
            import aiohttp

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            headers.update(self._api_extra_headers)

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self._api_timeout)) as session:
                async with session.post(self._api_base_url, json=payload, headers=headers) as resp:
                    text = await resp.text()
                    if resp.status >= 400:
                        raise RuntimeError(f"API status {resp.status}: {text}")
        except ImportError:
            import urllib.request

            def _blocking_call():
                req = urllib.request.Request(self._api_base_url)
                req.add_header("Authorization", f"Bearer {api_key}")
                req.add_header("Content-Type", "application/json")
                for k, v in self._api_extra_headers.items():
                    req.add_header(k, v)
                data = json.dumps(payload).encode("utf-8")
                with urllib.request.urlopen(req, data=data, timeout=self._api_timeout) as f:
                    return f.read().decode("utf-8")

            text = await asyncio.to_thread(_blocking_call)

        # Try parsing OpenAI-like response first
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and "choices" in obj:
                content = obj["choices"][0]["message"]["content"].strip()
            else:
                content = text.strip()
        except Exception:
            content = text.strip()

        # Extract JSON from content
        try:
            result = json.loads(content)
        except Exception:
            # Heuristic fallback
            lc = content.strip().lower()
            if "accept" in lc:
                return "accept", "ACCEPT"
            if "reject" in lc:
                return "reject", "REJECT"
            return "answer", content[:64]

        action = str(result.get("action", "")).lower()
        txt = str(result.get("content", "")).strip()
        return action, txt

    def _build_user_sim_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """Render a compact prompt for the user-simulator LLM.

        You can customize this template. Keep it concise to reduce latency.
        """
        import json

        convo = [
            {"role": m.get("role", ""), "content": m.get("content", "")} for m in messages
            if isinstance(m, dict)
        ]
        # Send only the last few turns to reduce token cost
        max_turns = int(self.config.get("max_history_turns", 6))
        convo = convo[-max_turns:]

        instruction = (
            "Based on the conversation so far, decide if you ACCEPT the assistant's answer,"
            " REJECT it, or ANSWER a clarifying question if the assistant asked one."
            " Return a strict JSON object: {\"action\": \"accept|reject|answer\", \"content\": \"...\"}."
        )
        return f"{instruction}\nConversation (JSON):\n{json.dumps(convo, ensure_ascii=False)}"
