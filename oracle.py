""" Python class for the verification oracle.

The class instantiate an oracle to check if the answers provided by the LLM are correct with respect to the
provided ground truth (prompt-expected answer pairs).
"""
import datetime


class AnswerVerificationOracle:
    def __init__(self):
        self.prompt_expected_answer_pairs = []
        self.results = []

    """ Adding the prompt-answer pairs.
    
    This method allows to add the prompt-expected answer pairs to the ground truth of the oracle.
    """
    def add_prompt_expected_answer_pair(self, prompt, expected_answer):
        """Add a prompt-expected answer pair to the oracle."""
        self.prompt_expected_answer_pairs.append((prompt, expected_answer))

    """ Verifying the answer correctness.

    This method checks whether the model's answer matches the expected answer for a given prompt.
    """
    def verify_answer(self, model_answer, prompt):
        result = {
            'prompt': prompt,
            'model_answer': model_answer,
            'expected_answer': None,
            'verification_result': None
        }

        for prompt_text, expected_answer in self.prompt_expected_answer_pairs:
            if prompt_text == prompt:
                result['expected_answer'] = expected_answer
                result['verification_result'] = False
                if ',' in expected_answer:
                    result['verification_result'] = True
                    for word in expected_answer.split(','):
                        if word.strip(' .,').lower() not in model_answer:
                            result['verification_result'] = False
                else:
                    if expected_answer.lower() in model_answer.lower():
                        result['verification_result'] = True
                
                """
                if result['verification_result'] == False:
                    print(f'\n++++++++++++\nRAG Answer: {model_answer}\nExpected Answer: {expected_answer}.\n++++++++++++')
                    human_feedback = input('t - True or f - False: ')
                    if human_feedback == 't': result['verification_result'] = True
                """
                break

        self.results.append(result)
        return result['verification_result']

    """ Writing the verification results to a file.

    This method produces in output the results of the validation procedure. 
    """
    def write_results_to_file(self,
                              file_path=f'tests/validation/results_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.txt'):
        total_results = len(self.results)
        correct_results = sum(int(result['verification_result']) for result in self.results)

        accuracy_percentage = (correct_results / total_results) * 100 if total_results > 0 else 0

        with open(file_path, 'w') as file:
            file.write(f"Overall Accuracy: {accuracy_percentage:.2f}%\n\n")

            for result in self.results:
                file.write(f"Prompt: {result['prompt']}\n")
                file.write(f"Model Answer: {result['model_answer']}\n")
                file.write(f"Expected Answer: {result['expected_answer']}\n")
                file.write(f"Verification Result: {result['verification_result']}\n")
                file.write("\n")
