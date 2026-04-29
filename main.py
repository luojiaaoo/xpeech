from xpeech.model import OpenAIChatModelWrapper
from xpeech.agent.agent import AgentWrapper, MessageHistoryCalls
import asyncer


message_history = []


async def main():
    aw = AgentWrapper(
        model_wrapper=OpenAIChatModelWrapper(
            base_url="https://ark.cn-beijing.volces.com/api/coding/v3",
            api_key="1cda8e1a-03e4-xxxx-d1a415985dbf",
            model_name="glm-5.1",
        ),
        deps_type=None,
        system_prompt="You are a helpful assistant.",
        message_history_calls=MessageHistoryCalls(
            get_message_history=lambda: message_history,
            set_message_history=lambda new_history: (
                message_history.clear,
                message_history.extend(new_history),
            ),
        ),
    )
    async for output in aw.run(
        user_prompt="What is the capital of France?", output_type=str
    ):
        print(output)
        ...

    async for output in aw.run(user_prompt="我刚刚说了什么", output_type=str):
        print(output)
        ...


asyncer.runnify(main)()
