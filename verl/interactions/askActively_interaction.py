import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4
from openai import OpenAI
from verl.interactions.base import BaseInteraction

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AskActivelyInteraction(BaseInteraction):
    """Multi-turn user simulator for PPO with an LLM-simulated user.

    Flow per dialogue:
    - Assistant either asks a clarifying question or gives an answer.
    - If clarifying question: user replies with a short answer; reward = -0.3; continue.
    - If final answer: user decides ACCEPT (+1, terminate) or REJECT (-1, continue).

    All user decisions are delegated to an external LLM API.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._instance: Dict[str, Dict[str, Any]] = {}

        # External LLM API config (vendor-agnostic HTTP schema)
        api_cfg = (config or {}).get("api", {})
        self._api_base_url: str = api_cfg.get("base_url", "")
        self._api_model: str = api_cfg.get("model", "")
        self._api_key: str = api_cfg.get("api_key", "USER_LLM_API_KEY")


    async def start_interaction(self, instance_id: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())

        # Optional context from dataset
        user_item = kwargs.get("user_item", None)
        # ground_truth = kwargs.get("ground_truth", None)


        self._instance[instance_id] = {
            "num_turns": 0,
            "reward": 0.0,
            "num_rejects": 0,
            "user_item": user_item,
        }
        return instance_id

    async def generate_response(
        self, instance_id: str, messages: List[Dict[str, Any]], **kwargs
    ) -> Tuple[bool, str, float, Dict[str, Any]]:
        
        user = OpenAI(
            api_key=self._api_key,
            base_url=self._api_base_url,
        )
        SYS_PROMPT_USER1 = """
        You will play the role of a user seeking help from an assistant. 
        Your goal is to have the assistant successfully solve the initial task you provide. 
        Below are the attributes and background of the problem you want help with:
        """

        SYS_PROMPT_USER2 = """
        Please follow these interaction rules carefully:

        1. You are acting purely as the user — not as the assistant.
        2. If the assistant determines your question is unclear, it will ask one clarifying question with several options.
        - Choose exactly **one** option that best matches your intended meaning and reply with its letter (e.g., “A”).
        3. If none of the provided options fit your situation, you may reply with:
        - “Either is fine.” (if multiple options could work), or a similar expression
        4. When the assistant gives a final answer, decide whether it correctly solves your original problem:
        - If you are satisfied, reply **exactly** with `"ACCEPT"`.
        - If you are not satisfied, reply **exactly** with `"REJECT"`.
        5. You should only respond to the assistant’s clarifying questions or its final answers — do not start new topics.
        6. Remember: your role is to represent the **user’s intent** faithfully and consistently throughout the conversation.
        """
        SYS_PROMPT_USER = SYS_PROMPT_USER1 + f"\n{kwargs.get('user_item', '')}\n" + SYS_PROMPT_USER2

        response_text = ""

        max_retries = 5
        retries = 0
        delay = 1  # 初始延迟时间（秒）
        for attempt in range(max_retries):
            try:
                response = user.chat.completions.create(
                    model = self._api_model,
                    messages = [
                        {"role": "system", "content": SYS_PROMPT_USER},
                        *messages, 
                        {"role": "user", "content": "Now continue as USER."}
                    ],
                    stream = False
                )
                response_text = response.choices[0].message.content
                break  # 成功则跳出重试循环
            except Exception as e:
                retries += 1
                if retries >= max_retries:
                    print(f"第 {attempt + 1} 次重试失败：{e}")
                    raise e  # 超过最大重试次数，抛出异常
                time.sleep(delay)  # 等待后重试
                delay *= 2  # 指数退避，每次等待时间翻倍

        self._instance[instance_id]["num_turns"] += 1
        reward = 0
        should_terminate = False

        if response_text.strip().upper() == "ACCEPT":
            should_terminate = True
            reward = 1.0
        elif response_text.strip().upper() == "REJECT":
            reward = -1.0
            
        reward += -0.3
        self._instance[instance_id]["reward"] = reward  
        return should_terminate, response_text, float(reward), {}

    async def calculate_score(self, instance_id: str, **kwargs) -> float:
        # Return the last user-turn reward to the trainer when needed
        return float(self._instance.get(instance_id, {}).get("reward", 0.0))

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance:
            del self._instance[instance_id]

    