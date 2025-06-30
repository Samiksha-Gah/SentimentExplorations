from llama_scorer import LLaMa3Scorer, SentimentScore
import unittest


class TestLLaMa3Scorer(unittest.TestCase):
    def setUp(self):
        self.scorer = LLaMa3Scorer(
            "llama3.2",
            "You are a financial expert assigning sentiment scores for news articles on continuous scale from -1 to 1, with -1 being bearish and 1 being bullish for the company. Return a score for the following article in JSON format:\n",
        )

    def test_get_response(self):
        response = self.scorer.get_response("Haidar’s Hedge Fund Loses 33% With Assets Plunging by $4 Billion")
        print(response)

    def test_get_sentiment(self):
        response = self.scorer.get_sentiment(headline="More sops needed to boost electronic manufacturing: Top govt official More sops needed to boost electronic manufacturing: Top govt official.  More sops needed to boost electronic manufacturing: Top govt official More sops needed to boost electronic manufacturing: Top govt official",source="The Economic Times India",summary="NEW DELHI | CHENNAI: India may have to offer electronic manufacturers additional sops such as cheap credit and incentives for export along with infrastructure support in order to boost production and help the sector compete with China, Vietnam and Thailand, according to a top government official.These incentives, over and above the proposed reduction of corporate tax to 15% for new manufacturing units, are vital for India to successfully attract companies looking to relocate manufacturing facilities.“While the tax announcements made last week send a very good signal, in order to help attract investments, we will need additional initiatives,” the official told ET, pointing out that Indian electronic manufacturers incur 8-10% higher costs compared with other Asian countries.Sops that are similar to the incentives for export under the existing Merchandise Exports from India Scheme (MEIS) are what the industry requires, the person said.MEIS gives tax credit in the range of 2-5%. An interest subvention scheme for cheaper loans and a credit guarantee scheme for plant and machinery are some other possible measures that will help the industry, the official added.“This should be 2.0 (second) version of the electronic manufacturing cluster (EMC) scheme, which is aimed at creating an ecosystem with an anchor company plus its suppliers to operate in the same area,” he said.Last week, finance minister Nirmala Sitharaman announced a series of measures to boost economic growth including a scheme allowing any new manufacturing company incorporated on or after October 1, to pay income tax at 15% provided the company does not avail of any other exemption or incentives.")
        print(response)


if __name__ == "__main__":
    unittest.main()