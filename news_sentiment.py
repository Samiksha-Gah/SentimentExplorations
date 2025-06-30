from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List

from pandas import DataFrame

from dotenv import load_dotenv
import os 


class FinnhubFetcher:
    """Helper class for fetching finnhub.io data"""

    def __init__(self):
        load_dotenv()
        self.client = finnhub.Client(api_key=os.getenv("api_key"))

    def fetch_company_news(
        self, symbol: str, from_date: str, to_date: str
    ) -> List[Dict]:
        """Fetch /company-news API output from Finnhub

        Args:
            symbol (str): company ticker symbolk
            from_date (str): start date for news collection period in UTC format (%Y-%m-%d)
            to_date (str): end date for news collection period in UTC format (%Y-%m-%d)

        Returns:
            List[Dict]: list of company news data with keys "category", "datetime", "headline", "id", "image", "related", "source", "summary", "url"
        """

        return self.client.company_news(symbol, _from=from_date, to=to_date)

    def dict_to_dataframe(self, news_dict: Dict) -> DataFrame:
        """Converts company news dictionary data into pandas DataFrame

        Args:
            news_dict (Dict): company news data with keys "category", "datetime", "headline", "id", "image", "related", "source", "summary", "url"

        Returns:
            DataFrame: company news table with ticker symbol as an index and "date", "headline", "source", "summary", and "url" as columns
        """
        
        df = DataFrame(news_dict, columns=["datetime", "headline", "source", "summary", "url"])
        df.rename(columns={"datetime": "date"}, inplace=True)
        return df


class SentimentScorer(ABC):
    """Abstract class for language-model based sentiment scorers"""

    @abstractmethod
    def get_sentiment(
        self, headline: str, summary: str, source: str, contents: str = None
    ) -> float:
        """Generates sentiment using underlying language model

        Args:
            headline (str): article headline
            summary (str): article summary
            source (str): news publisher/source
            contents (str, optional): body text of article. Defaults to None.

        Returns:
            float: sentiment score ranging from -1 to 1
        """

    @abstractmethod
    def get_response(self, input_text: str) -> str:
        """Queries response from underlying language model

        Args:
            input_text (str): text fed into language model

        Returns:
            str: response from the language model
        """

class ArticleData:
    """Collects, processes, and stores news article data and sentiment scores"""

    def __init__(self, fetcher: FinnhubFetcher, scorers: List[SentimentScorer]):
        """
        Args:
            fetcher (FinnhubFetcher): Instance of FinnhubFetcher for fetching news data.
            scorers (List[SentimentScorer]): List of sentiment scorers for analyzing articles.
        """
        self.fetcher = fetcher
        self.scorers = scorers
        self.data = DataFrame()

    def collect_data(self, from_date: str, to_date: str):
        """Collects news article data and sentiment scores

        Args:
            from_date (str): start date for news collection period in UTC format (%Y-%m-%d)
            to_date (str): end date for news collection period in UTC format (%Y-%m-%d)
        """
        articles = []

        for symbol in self.fetcher.symbol_list:
            news = self.fetcher.fetch_company_news(symbol, from_date, to_date)
            df = self.fetcher.dict_to_dataframe(news)

            for scorer in self.scorers:
                df[f'sentiment_{scorer.__class__.__name__}'] = df.apply(
                    lambda row: scorer.get_sentiment(
                        headline=row['headline'],
                        summary=row['summary'],
                        source=row['source']
                    ),
                    axis=1
                )

            articles.append(df)

        self.data = concat(articles, ignore_index=True)

    def aggregate_sentiment(self, sentiments: List[float]) -> float:
        """Combines sentiment scores for multiple articles in a single day

        Args:
            sentiments (List[float]): sentiment scores from particular model corresponding to different articles in a day

        Returns:
            float: aggregated sentiment score
        """
        return sum(sentiments) / len(sentiments) if sentiments else 0.0

    def get_volume(self, date: str, symbol: str) -> int:
        """Counts the volume (number of articles) for a given day and company

        Args:
            date (str): date in UTC format (%Y-%m-%d)
            symbol (str): ticker symbol for company

        Returns:
            int: article volume for the given day and company
        """
        return len(self.data[(self.data['date'] == date) & (self.data.index == symbol)])

    def process_data(self) -> None:
        """Data processing step for cleaning/feature engineering, etc."""

    def to_csv(self, path: str | Path, separator: str = ",") -> None:
        """Writes data to CSV format

        Args:
            path (str | Path): Path to save the CSV file.
            separator (str, optional): Delimiter for the CSV file. Defaults to ",".
        """
        if not self.data.empty:
            self.data.to_csv(path, sep=separator, index=False)
