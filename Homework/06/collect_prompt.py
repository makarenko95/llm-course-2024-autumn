def generate(sample, answer, add_full_example, is_value):
    s = f"The following are multiple choice questions (with answers) about {sample['subject']}.\n"
    choices = sample['choices']
    s += sample['question'] + "\n"
    alpha = "ABCD"
    for i in range(4):
        s += f"{alpha[i]}. {choices[i]}\n"
    s += "Answer:"
    if answer:
        idx = sample['answer']
        letter = chr(ord('A') + idx)
        value = letter if is_value else idx
        s += f" {value}"
        if add_full_example:
            s += f". {choices[idx]}"
    return s


def create_prompt(sample: dict) -> str:
    """
    Generates a prompt for a multiple choice question based on the given sample.

    Args:
        sample (dict): A dictionary containing the question, subject, choices, and answer index.

    Returns:
        str: A formatted string prompt for the multiple choice question.
    """

    return generate(sample, True, False, False)

def create_prompt_with_examples(sample: dict, examples: list, add_full_example: bool = False) -> str:
    """
    Generates a 5-shot prompt for a multiple choice question based on the given sample and examples.

    Args:
        sample (dict): A dictionary containing the question, subject, choices, and answer index.
        examples (list): A list of 5 example dictionaries from the dev set.
        add_full_example (bool): whether to add the full text of an answer option

    Returns:
        str: A formatted string prompt for the multiple choice question with 5 examples.
    """
    result = ""
    for question in examples:
        result += generate(question, True, add_full_example, True) + '\n\n'
    result += generate(sample, False, add_full_example, True)
    return result