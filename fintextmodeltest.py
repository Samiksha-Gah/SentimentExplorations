from fintextmodel import FinBERTScorer

if __name__ == "__main__":
    # Initialize the FinBERT scorer 
    scorer = FinBERTScorer()

    #Example inputs 
    headline = "The company announced significant plans to better its management team, but little impact."
    summary = "The announcement did not have a significant increase on the stock price."
    contents = None
    source = "Reuters"

    # Get sentiment analysis result
    result = scorer.get_sentiment(headline=headline, summary=summary, source=source, contents=contents)
    print(f"Sentiment Label: {result['label']}")
    print(f"Confidence Score: {result['score']:.2f}")
