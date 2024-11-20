import sys
sys.path.append("Homework/06")
import unittest
import re
from datasets import load_dataset

from collect_prompt import create_prompt, create_prompt_with_examples


class TestCreatePrompt(unittest.TestCase):
    def setUp(self):
        self.dataset = load_dataset("cais/mmlu", "medical_genetics")

    def test_create_prompt_format(self):
        pattern = re.compile(
            r"The following are multiple choice questions \(with answers\) about (.+?)\.\n"
            r"(.+?)\n"
            r"A\. (.+?)\n"
            r"B\. (.+?)\n"
            r"C\. (.+?)\n"
            r"D\. (.+?)\n"
            r"Answer:"
        )

        for i, example in enumerate(self.dataset['test']):
            prompt = create_prompt(example)
            self.assertRegex(prompt, pattern, f"Prompt does not match the expected format for example {i}")

    def test_create_prompt_few_shot_format(self):
        sample = {
            'question': 'DNA ligase is',
            'subject': 'medical_genetics',
            'choices': [
                'an enzyme that joins fragments in normal DNA replication',
                'an enzyme of bacterial origin which cuts DNA at defined base sequences',
                'an enzyme that facilitates transcription of specific genes',
                'an enzyme which limits the level to which a particular nutrient reaches'
            ],
            'answer': 0
        }

        examples = [
            {
                'question': 'Large triplet repeat expansions can be detected by:',
                'subject': 'medical_genetics',
                'choices': [
                    'polymerase chain reaction.',
                    'single strand conformational polymorphism analysis.',
                    'Southern blotting.',
                    'Western blotting.'
                ],
                'answer': 2
            },
            {
                'question': 'DNA ligase is',
                'subject': 'medical_genetics',
                'choices': [
                    'an enzyme that joins fragments in normal DNA replication',
                    'an enzyme of bacterial origin which cuts DNA at defined base sequences',
                    'an enzyme that facilitates transcription of specific genes',
                    'an enzyme which limits the level to which a particular nutrient reaches'
                ],
                'answer': 0
            },
            {
                'question': 'A gene showing codominance',
                'subject': 'medical_genetics',
                'choices': [
                    'has both alleles independently expressed in the heterozygote',
                    'has one allele dominant to the other',
                    'has alleles tightly linked on the same chromosome',
                    'has alleles expressed at the same time in development'
                ],
                'answer': 0
            },
            {
                'question': 'Which of the following conditions does not show multifactorial inheritance?',
                'subject': 'medical_genetics',
                'choices': [
                    'Pyloric stenosis',
                    'Schizophrenia',
                    'Spina bifida (neural tube defects)',
                    'Marfan syndrome'
                ],
                'answer': 3
            },
            {
                'question': 'The stage of meiosis in which chromosomes pair and cross over is:',
                'subject': 'medical_genetics',
                'choices': [
                    'prophase I',
                    'metaphase I',
                    'prophase II',
                    'metaphase II'
                ],
                'answer': 0
            }
        ]

        # Call the function
        result = create_prompt_with_examples(sample, examples)

        # Define the regex pattern
        pattern = (
            r"(The following are multiple choice questions \(with answers\) about medical_genetics\.\n"
            r".+\n"
            r"A\. .+\n"
            r"B\. .+\n"
            r"C\. .+\n"
            r"D\. .+\n"
            r"Answer: [ABCD]\n\n){5}"
            r"The following are multiple choice questions \(with answers\) about medical_genetics\.\n"
            r"DNA ligase is\n"
            r"A\. an enzyme that joins fragments in normal DNA replication\n"
            r"B\. an enzyme of bacterial origin which cuts DNA at defined base sequences\n"
            r"C\. an enzyme that facilitates transcription of specific genes\n"
            r"D\. an enzyme which limits the level to which a particular nutrient reaches\n"
            r"Answer:$"
        )

        # Check if the result matches the pattern
        self.assertRegex(result, pattern)


if __name__ == '__main__':
    unittest.main()
