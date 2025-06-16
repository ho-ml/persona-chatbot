import gc
import torch

from typing import List, Dict
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from config import ChatBotConfig

class ChatBot:
    def __init__(self, config):
        # config
        self.config = config

        # sampling parameters
        self.sampling_params = SamplingParams(
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stop=["<|eot_id|>"]
        )
        
        # initialize
        self._initialize()

    def infer(self, query: str, history: List[Dict]):
        messages = history.copy()

        messages = self._remove_context(messages)

        messages.append({"role": "user", "content": query})
        context = self._format_context(query)
        messages.append({"role": "user", "content": context})

        prompt = self._process_prompt(messages)

        responses = self.llm.generate(prompt, self.sampling_params)
        response = responses[0].outputs[0].text.strip()
        messages.append({"role": "assistant", "content": response})

        # multi-turn
        samples = []
        i = 0

        while i < len(messages):
            if messages[i]["role"] == "user":
                user_msg = messages[i]["content"]
                i += 1

                if i < len(messages) and messages[i]["role"] == "user" and messages[i]["content"].startswith("<context>"):
                    i += 1

                if i < len(messages) and messages[i]["role"] == "assistant":
                    assist_msg = messages[i]["content"]
                    i += 1
                    samples.append((user_msg, assist_msg))
                else:
                    samples.append((user_msg, ""))
            
            else:
                i += 1

        return samples, messages

    def _initialize(self):
        # cache 
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()

        # model
        self.llm = LLM(model=self.config.model, max_model_len=self.config.max_model_len)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model)

        # retriever
        self.embed_model = OpenAIEmbeddings(api_key=self.config.api_key)
        self.vector_db = Chroma(
            embedding_function=self.embed_model, 
            persist_directory=self.config.vector_db
        )

    def _remove_context(self, messages: List[Dict]):
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "user" and messages[i]["content"].startswith("<context>"):
                return messages[:i] + messages[i+1:]
        
        return messages
    
    def _format_context(self, query: str):
        docs = self.vector_db.similarity_search(query, k=self.config.k)

        context = "<context>\n"
        for i, doc in enumerate(docs, start=1):
            context += f"<doc{i}>{doc.page_content}</doc{i}>\n"
        context += "</context>"

        return context
    
    def _process_prompt(self, messages: List[Dict]):
        # build prompt based on LLaMA3
        system_header = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        prompt = system_header + self.config.system_prompt.strip() + "<|eot_id|>"

        for msg in messages:
            role = msg["role"]
            content = msg["content"].strip()
            prompt += f"<|start_header_id|>{role}<|end_header_id|>\n{content}<|eot_id|>" 
        
        return prompt

if __name__ == "__main__":
    # instance
    config = ChatBotConfig()
    chatbot = ChatBot(config=config)

    # example query
    query = "아저씨는 혼자일 때 어떤 생각을 해?"
    histories = []

    # inference
    _, history = chatbot.infer(query, histories)
    response = history[-1]['content']

    print(f"query:\n{query}")
    print(f"response:\n{response}")
