from news_sentiment import SentimentScorer
from ollama import chat, ChatResponse
from pydantic import BaseModel


class SentimentScore(BaseModel):
    score: float


class LLaMa3Scorer(SentimentScorer):
    def __init__(self, model_name: str, prompt: str):
        self.model_name = model_name
        self.prompt = prompt

    def get_response(self, input_text: str):
        message = f"{self.prompt}{input_text}"
        print(message)
        response = chat(
            model=self.model_name,
            messages=[{"role": "user", "content": message}],
            format=SentimentScore.model_json_schema(),
            options={"temperature": 0},
        )
        content = SentimentScore.model_validate_json(response.message.content)
        return content

    def get_sentiment(self, headline: str, summary: str, source: str, contents=None):
        input_text = f"""Article Headline: {headline}
Article Summary: {summary}
Article Source: {source}
"""
        return self.get_response(input_text)
