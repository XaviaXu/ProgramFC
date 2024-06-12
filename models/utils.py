import time

import backoff  # for exponential backoff
import openai
from openai import OpenAI
from together import Together
import os
import asyncio
from typing import Any, Type

from together.error import InvalidRequestError


# @backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def completions_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)

# @backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def chat_completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

async def dispatch_openai_prompt_requests(
    messages_list: list[list[dict[str,Any]]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    stop_words: list[str]
) -> list[str]:
    async_responses = [
        openai.Completion.acreate(
            model=model,
            prompt=x,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty = 0.0,
            presence_penalty = 0.0,
            stop = stop_words
        )
        for x in messages_list
    ]
    return await asyncio.gather(*async_responses)

class OpenAIModel:
    def __init__(self, API_KEY, model_name, stop_words, max_new_tokens) -> None:
        openai.api_key = API_KEY
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.stop_words = stop_words
        if 'gpt' in model_name:
            self.openai = OpenAI(api_key=API_KEY)
        else:
            self.openai = Together(api_key=API_KEY)

    # used for chat-gpt and gpt-4
    def chat_generate(self, input_string,chatcot, adaptive,temperature = 0.2):
        if(chatcot or adaptive):
            message = input_string
        else:
            message = [
                        {"role": "user", "content": input_string}
                    ]
        while True:
            try:
                response = self.openai.chat.completions.create(
                        model = self.model_name,
                        messages=message,
                        max_tokens = self.max_new_tokens,
                        temperature = temperature,
                        top_p = 1.0,
                        stop = self.stop_words
                )
                break
            except Exception as e:
                # print(e)
                if 'Input validation error' in e.args[0]:
                    del message[-3:-1]
                # time.sleep(5)
        generated_text = response.choices[0].message.content.strip()
        return generated_text
    
    # used for text/code-davinci

    def convert_prompt(self,messages):
        prompt_list = ['<|begin_of_text|>']
        for message in messages:
            prompt_list.append(f"<|start_header_id|>{message['role']}<|end_header_id|>\n{message['content']}<|eot_id|>\n")
        prompt_list.append("<|start_header_id|>assistant<|end_header_id|>")
        return prompt_list

    def prompt_generate(self, input_string, adaptive,temperature = 0.0):
        if adaptive:
            prompt_list = self.convert_prompt(input_string)
        while True:
            try:
                if adaptive:
                    prompt = '\n'.join(prompt_list)
                else:
                    prompt = input_string
                response = self.openai.completions.create(
                    model=self.model_name,
                    prompt=prompt,
                    max_tokens=self.max_new_tokens,
                    temperature=temperature,
                    top_p=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    stop=self.stop_words
                )
                break
            except Exception as e:
                # print(e)
                if 'Input validation error' in e.args[0]:
                    del prompt_list[-4:-2]

        generated_text = response.choices[0].text.strip()
        return generated_text

    def generate(self, input_string, chatcot,adaptive, temperature = 0.0):
        if self.model_name in ['meta-llama/Meta-Llama-3-70B']:
            return self.prompt_generate(input_string, adaptive,temperature)
        else:
            return self.chat_generate(input_string,chatcot, adaptive,temperature)
        # else:
        #     raise Exception("Model name not recognized")
    
    def batch_chat_generate(self, messages_list, temperature = 0.2):
        open_ai_messages_list = []
        for message in messages_list:
            open_ai_messages_list.append(
                [{"role": "user", "content": message}]
            )
        predictions = asyncio.run(
            dispatch_openai_chat_requests(
                    open_ai_messages_list, self.model_name, temperature, self.max_new_tokens, 1.0, self.stop_words
            )
        )
        return [x.choices[0].message.content.strip() for x in predictions]
    
    def batch_prompt_generate(self, prompt_list, temperature = 0.0):
        predictions = asyncio.run(
            dispatch_openai_prompt_requests(
                    prompt_list, self.model_name, temperature, self.max_new_tokens, 1.0, self.stop_words
            )
        )
        return [x['choices'][0]['text'].strip() for x in predictions]

    def batch_generate(self, messages_list, temperature = 0.0):
        if self.model_name in ['text-davinci-002', 'code-davinci-002', 'text-davinci-003']:
            return self.batch_prompt_generate(messages_list, temperature)
        elif self.model_name in ['gpt-4', 'gpt-3.5-turbo']:
            return self.batch_chat_generate(messages_list, temperature)
        else:
            raise Exception("Model name not recognized")

    def generate_insertion(self, input_string, suffix, temperature = 0.0):
        response = completions_with_backoff(
            model = self.model_name,
            prompt = input_string,
            suffix= suffix,
            temperature = temperature,
            max_tokens = self.max_new_tokens,
            top_p = 1.0,
            frequency_penalty = 0.0,
            presence_penalty = 0.0
        )
        generated_text = response['choices'][0]['text'].strip()
        return generated_text