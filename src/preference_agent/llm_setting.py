__author__ = "Dong Yihan"

import openai
import tiktoken
from openai import OpenAI


class OPENAI:

    def __init__(self):
        self.status = {
            "status": 500,
            "message": "error in initialization of class OPENAI"
        }

        self.openai_key = ""

        self.gpt_config = {
            'model_name': 'gpt-4.1',
        }

        # self.encoding = tiktoken.encoding_for_model(self.gpt_config["model_name"])  # 4o version
        self.encoding = tiktoken.get_encoding("cl100k_base")

        self.status.update(status=200, message="successfully initialize class openai")
        return

    def get_gpt_response(self, prompts: list[dict[str, str]]):
        self.status.update(status=500, message="error in get_gpt_response")

        openai.api_key = self.openai_key
        response = openai.chat.completions.create(
            model=self.gpt_config["model_name"],
            messages=prompts,
            temperature=0
        )

        response_content = response.choices[0].message.content

        self.status.update(status=200, message="successfully get responses")
        return response_content
