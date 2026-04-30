from xpeech.agent.model import OpenAIChatModelWrapper
from xpeech.agent.agent import AgentWrapper, MessageHistoryCalls
from asyncer import runnify
from settings import settings


message_history = ""


def set_message_history(new_history: str):
    global message_history
    message_history = new_history


async def main():

    aw = AgentWrapper(
        model_wrapper=OpenAIChatModelWrapper(
            base_url=settings.base_url,
            api_key=settings.api_key,
            model_name=settings.model_name,
        ),
        deps_type=None,
        system_prompt="You are a helpful assistant.",
        message_history_calls=MessageHistoryCalls(
            get_message_history=lambda: message_history,
            set_message_history=set_message_history,
        ),
        workspace="./workspace/luoja",
    )
    async for output in aw.run(user_prompt="创建一个叫a.txt的文件，如果有了则不用操作", output_type=str):
        # print(output)
        ...

    async for output in aw.run(user_prompt="往文件里追加123", output_type=str):
        # print(output)
        ...

    async for output in aw.run(user_prompt="把a.txt复制到sub/b.txt", output_type=str):
        # print(output)
        ...

    async for output in aw.run(user_prompt="删除a.txt", output_type=str):
        # print(output)
        ...

    async for output in aw.run(user_prompt="说个笑话", output_type=str):
        # print(output)
        ...
    for i in range(10):
        async for output in aw.run(user_prompt="再说个笑话", output_type=str):
            # print(output)
            ...


runnify(main)()
