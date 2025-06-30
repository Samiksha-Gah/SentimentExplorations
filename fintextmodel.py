from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

class FinBERTScorer:
    """
    A sentiment scorer class using FinBERT for financial text sentiment analysis
    """
    def __init__(self):
        # Specify the FinBERT model name
        model_name = "yiyanghkust/finbert-tone"

        # Load the tokenizer and model from Hugging Face
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.nlp = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)
            print("FinBERT model and tokenizer loaded successfully.")
        except Exception as e:
            print(f"Error loading FinBERT model or tokenizer: {e}")
            raise

    def get_sentiment(self, headline: str, summary: str, source: str, contents: str = None):
        """
        Generates sentiment using the FinBERT model.

        Args:
            headline (str): article headline
            summary (str): article summary
            source (str): news publisher/source
            contents (str, optional): body text of the article. Defaults to None.

        Returns:
            dict: dictionary with "label" (sentiment classification) and "score" (confidence score)
        """
        text_to_analyze = f"{headline} {summary}"
        if contents:
            text_to_analyze += f" {contents}"

        # Perform sentiment analysis
        result = self.nlp(text_to_analyze)[0]  # Get the first result
        return {"label": result["label"].lower(), "score": result["score"]}
