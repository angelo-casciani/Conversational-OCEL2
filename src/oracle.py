import datetime
import os
from sklearn.metrics import (
    matthews_corrcoef, roc_auc_score, accuracy_score,
    precision_score, recall_score, f1_score
)


class AnswerVerificationOracle:
    def __init__(self, info_run):
        self.question_expected_answer_pairs = {}
        self.positives = 0
        self.negatives = 0
        self.true_positives = 0
        self.true_negatives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.accuracy = 0.0
        self.precision = 0.0
        self.recall = 0.0
        self.f1score = 0.0
        self.mcc = 0.0
        self.auc = 0.0
        self.results = []
        self.run_info = info_run

    def add_question_expected_answer_pair(self, question, expected_answer):
        self.question_expected_answer_pairs[question] = expected_answer
        if expected_answer == "True":
            self.positives += 1
        elif expected_answer == "False":
            self.negatives += 1

    def verify_answer(self, prompt, question, model_answer):
        result = {
            'prompt': prompt,
            'question': question,
            'model_answer': model_answer,
            'expected_answer': None,
            'predicted_answer': None,
            'verification_result': False
        }
        expected_answer = self.question_expected_answer_pairs.get(question)
        if expected_answer is not None:
            result['expected_answer'] = expected_answer
            normalized_answer = (model_answer.strip().split()[-1].strip(' .,:;!?').capitalize())
            if normalized_answer in ["True", "False"]:
                result['predicted_answer'] = normalized_answer
                result['verification_result'] = (normalized_answer == expected_answer)
            else:
                result['predicted_answer'] = "UNKNOWN"
                result['verification_result'] = False

        print(
            f"\nPrompt: {prompt}\n"
            f"Answer: {model_answer}\n"
            f"Predicted Answer: {result['predicted_answer']}\n"
            f"Expected Answer: {expected_answer}\n"
            f"Result: {result['verification_result']}"
        )
        self.results.append(result)
        return result['verification_result']

    def compute_stats(self):
        y_true = []
        y_pred = []
        self.true_positives = 0
        self.true_negatives = 0
        self.false_positives = 0
        self.false_negatives = 0

        for result in self.results:
            expected_label = 1 if result['expected_answer'] == "True" else 0
            predicted_label = 1 if result['predicted_answer'] == "True" else 0
            y_true.append(expected_label)
            y_pred.append(predicted_label)

            if predicted_label == 1 and expected_label == 1:
                self.true_positives += 1
            elif predicted_label == 0 and expected_label == 0:
                self.true_negatives += 1
            elif predicted_label == 1 and expected_label == 0:
                self.false_positives += 1
            elif predicted_label == 0 and expected_label == 1:
                self.false_negatives += 1

        if y_true:
            self.accuracy = accuracy_score(y_true, y_pred)
            self.precision = precision_score(y_true, y_pred, zero_division=0)
            self.recall = recall_score(y_true, y_pred, zero_division=0)
            self.f1score = f1_score(y_true, y_pred, zero_division=0)
            self.mcc = matthews_corrcoef(y_true, y_pred)
            if len(set(y_true)) > 1:
                self.auc = roc_auc_score(y_true, y_pred)
            else:
                self.auc = 0.5  # Undefined AUC fallback
        else:
            self.accuracy = self.precision = self.recall = self.f1score = 0.0
            self.mcc = 0.0
            self.auc = 0.5

    def write_results_to_file(self):
        file_path = os.path.join(
            os.path.dirname(__file__), "..", "tests", "validation",
            f"results_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
        )
        self.compute_stats()

        with open(file_path, "w") as file:
            file.write("INFORMATION ON THE RUN\n\n")
            for key, value in self.run_info.items():
                file.write(f"{key}: {value}\n")

            file.write("\n-----------------------------------\n")
            file.write("PERFORMANCE METRICS\n")
            file.write("-----------------------------------\n")
            file.write(f"Accuracy: {self.accuracy:.2%}\n")
            file.write(f"Precision: {self.precision:.2%}\n")
            file.write(f"Recall: {self.recall:.2%}\n")
            file.write(f"F1-Score: {self.f1score:.2f}\n")
            file.write(f"Matthews Corr. Coeff. (MCC): {self.mcc:.2f}\n")
            file.write(f"Area Under Curve (AUC): {self.auc:.2f}\n\n")

            file.write("-----------------------------------\n")
            file.write("CONFUSION MATRIX\n")
            file.write("-----------------------------------\n")
            file.write(f"Total Questions: {len(self.results)}\n")
            file.write(f"Positive Labels: {self.positives}\n")
            file.write(f"Negative Labels: {self.negatives}\n\n")
            file.write(f"True Positives (TP): {self.true_positives}\n")
            file.write(f"True Negatives (TN): {self.true_negatives}\n")
            file.write(f"False Positives (FP): {self.false_positives}\n")
            file.write(f"False Negatives (FN): {self.false_negatives}\n\n")

            file.write("-----------------------------------\n")
            file.write("DETAILED RESULTS\n")
            file.write("-----------------------------------\n\n")

            for result in self.results:
                file.write(f"Prompt: {result['prompt']}\n")
                file.write(f"Model Answer: {result['model_answer']}\n")
                file.write(f"Predicted Answer: {result['predicted_answer']}\n")
                file.write(f"Expected Answer: {result['expected_answer']}\n")
                file.write(
                    f"Result: "
                    f"{'CORRECT' if result['verification_result'] else 'INCORRECT'}\n"
                )
                file.write("\n#####################################################################################\n")
