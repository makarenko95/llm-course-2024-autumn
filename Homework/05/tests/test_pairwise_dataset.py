from unittest import TestCase
from scripts.pairwise_dataset import IMDBPairwiseDataset
import warnings
warnings.filterwarnings("ignore")

from transformers import AutoTokenizer
from datasets import load_dataset


class TestPairwiseDataset(TestCase):

    def setUp(self):
        self.reward_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")

    def test_simple(self):
        test_dataset = [
            {'text': '1 2 3iwejgiwojgajej', 'label': 1},
            {'text': '2', 'label': 1},
            {'text': '3', 'label': 1},
            {'text': '4', 'label': 1},
            {'text': '5', 'label': 1},
            {'text': 'a', 'label': 0},
            {'text': 'b', 'label': 0},
            {'text': 'c', 'label': 0},
            {'text': 'd', 'label': 0},
            {'text': 'e', 'label': 0},
        ]

        dataset = IMDBPairwiseDataset(test_dataset, self.reward_tokenizer, accepted_label=1)
        self.assertEqual(len(dataset), 25)

        sample = dataset[0]
        self.assertEqual(self.reward_tokenizer.decode(sample['input_ids_chosen'][1:-1]), test_dataset[0]['text'])
        self.assertEqual(self.reward_tokenizer.decode(sample['input_ids_rejected'][1:-1]), test_dataset[5]['text'])

        sample = dataset[1]
        self.assertEqual(self.reward_tokenizer.decode(sample['input_ids_chosen'][1:-1]), test_dataset[0]['text'])
        self.assertEqual(self.reward_tokenizer.decode(sample['input_ids_rejected'][1:-1]), test_dataset[6]['text'])

        sample = dataset[-1]
        self.assertEqual(self.reward_tokenizer.decode(sample['input_ids_chosen'][1:-1]), test_dataset[4]['text'])
        self.assertEqual(self.reward_tokenizer.decode(sample['input_ids_rejected'][1:-1]), test_dataset[9]['text'])

    def test_imdb(self):

        TARGET_LABEL = 0   # negative reviews
        imdb = load_dataset("imdb", split='train')
        reward_data = IMDBPairwiseDataset(imdb, self.reward_tokenizer, accepted_label=TARGET_LABEL)
        self.assertEqual(len(reward_data), 12500 * 12500)

        sample = reward_data[31337]
        self.assertEqual(self.reward_tokenizer.decode(sample['input_ids_chosen'][1:-1]).replace(' ', ''), imdb[2]['text'].replace(' ', ''))
        self.assertEqual(self.reward_tokenizer.decode(sample['input_ids_rejected'][1:-1]).replace(' ', ''), imdb[18837]['text'].replace(' ', ''))