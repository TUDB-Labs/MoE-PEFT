
# dataset_hf_path = "UBC-NLP/palmx_2025_subtask2_islamic"
dataset_hf_path = "UBC-NLP/palmx_2025_subtask1_culture"

import sys
from datasets import load_dataset

train_data = load_dataset(dataset_hf_path, split = 'train')

adapter_path = "/export/home/hhassan/MoE-PEFT/casual_0"
model_name = "QCRI/Fanar-1-9B-Instruct"

import torch
import numpy as np
from datasets import load_dataset
import pandas as pd
import moe_peft
from moe_peft.modules import LLMBatchConfig, LLMModelInput


DEFAULT_PROMPT_TEMPLATE = "{}\n\n{}\nالجواب:"
DEFAULT_CHOICE_PREFIXES = ["A.", "B.", "C.", "D."]

class MCQProcessor:
    """
    Handles processing and evaluation of Multiple Choice Questions (MCQs)
    using a MoE-PEFT Causal Language Model.
    """
    def __init__(self, model_name, device=None,
                 prompt_template=DEFAULT_PROMPT_TEMPLATE,
                 choice_prefixes=None):

        if device:
            self.device = device
        else:
            self.device = moe_peft.backend.default_device_name()
            print(f"Using device: {self.device}")

        self.prompt_template = prompt_template
        self.choice_prefixes = choice_prefixes if choice_prefixes is not None else DEFAULT_CHOICE_PREFIXES

        print(f"Loading model and tokenizer: {model_name}...")
        try:
            # Load MoE model
            self.model = moe_peft.LLMModel.from_pretrained(
                model_name,
                device=self.device,
                # bits=4,  # Use 4-bit quantization
                load_dtype=torch.bfloat16,
                attn_impl="eager"
            )
            
            # Load tokenizer
            self.tokenizer = moe_peft.Tokenizer(model_name)
            
            # Load adapter
            self.adapter_name = self.model.load_adapter(adapter_path)
            print(f"Loaded adapter: {self.adapter_name}")
            
        except Exception as e:
            print(f"Error loading model/tokenizer for {model_name}: {e}")
            print("If using a gated model (e.g., Llama), ensure you have access and are logged in via `huggingface-cli login`.")
            print("Or, try a different model like 'distilgpt2' or 'gpt2'.")
            raise

        print("Model and tokenizer loaded successfully.")

    def _format_doc_to_prompt_text(self, doc):
        """
        Formats a document (MCQ item) into a full prompt string.
        """
        question_text = doc["question"]

        options = []
        for i, opt_text in enumerate(doc["choices"]):
            if i < len(self.choice_prefixes):
                options.append(f"{self.choice_prefixes[i]} {opt_text}")
            else:
                options.append(f"{i+1}. {opt_text}")

        return self.prompt_template.format(question_text, "\n".join(options))

    def _get_choice_labels(self, doc):
        """
        Returns the single letter labels for choices (e.g., ["A", "B", "C"]).
        """
        num_choices = len(doc["choices"])
        return [self.choice_prefixes[i][0] for i in range(num_choices) if i < len(self.choice_prefixes)]

    def process_batch(self, batch_docs):
        """
        Processes a batch of MCQ documents.
        """
        flat_full_sequences = []
        flat_prompt_tokens_ns_map = []
        flat_continuation_tokens_ns_map = []
        doc_choice_counts = []

        for doc in batch_docs:
            prompt_text = self._format_doc_to_prompt_text(doc)
            choice_labels = self._get_choice_labels(doc)
            doc_choice_counts.append(len(choice_labels))

            prompt_tokens_ns = self.tokenizer.encode(prompt_text, add_special_tokens=False)

            for choice_label in choice_labels:
                continuation_text = " " + choice_label
                full_sequence_text = prompt_text + continuation_text
                flat_full_sequences.append(full_sequence_text)
                flat_prompt_tokens_ns_map.append(prompt_tokens_ns)
                continuation_tokens_ns = self.tokenizer.encode(continuation_text, add_special_tokens=False)
                flat_continuation_tokens_ns_map.append(continuation_tokens_ns)

        if not flat_full_sequences:
            return [], [], []

        # Encode all sequences with max length limit
        max_seq_len = 512
        print(f"Encoding {len(flat_full_sequences)} sequences with max length {max_seq_len}...")
        all_tokens = []
        max_len = 0
        for seq_text in flat_full_sequences:
            tokens = self.tokenizer.encode(seq_text, add_special_tokens=True)
            # Truncate if too long
            if len(tokens) > max_seq_len:
                tokens = tokens[:max_seq_len]
            all_tokens.append(tokens)
            max_len = max(max_len, len(tokens))

        # Pad sequences to the same length
        padded_tokens = []
        for tokens in all_tokens:
            if len(tokens) < max_len:
                # Pad with pad tokens
                padded = tokens + [self.tokenizer.pad_id_] * (max_len - len(tokens))
            else:
                padded = tokens
            padded_tokens.append(padded)

        # Create batch configuration for MoE model
        batch_config = [LLMBatchConfig(
            adapter_name_=self.adapter_name,
            batch_start_idx_=0,
            batch_end_idx_=len(padded_tokens)
        )]

        # Create model input
        model_input = LLMModelInput(
            batch_configs_=batch_config,
            batch_tokens_=padded_tokens,
            inference_mode_=True
        )

        with torch.no_grad():
            outputs = self.model.forward(model_input)
            # Get the logits from the first (and only) output
            logits_batch = outputs[0].logits

        flat_choice_log_likelihoods = []
        for b_idx in range(len(flat_full_sequences)):
            current_logits_slice = logits_batch[b_idx]
            current_prompt_tokens_ns = flat_prompt_tokens_ns_map[b_idx]
            current_continuation_tokens_ns = flat_continuation_tokens_ns_map[b_idx]

            if not current_continuation_tokens_ns:
                flat_choice_log_likelihoods.append(-float('inf'))
                continue

            # Calculate start position of continuation tokens
            # Add 1 if BOS token is present
            idx_offset = 1 if self.tokenizer.bos_id_ is not None else 0
            start_of_continuation_in_ids = idx_offset + len(current_prompt_tokens_ns)
            
            # Find actual sequence length (without padding)
            actual_tokens = all_tokens[b_idx]
            actual_sequence_len = len(actual_tokens)

            if start_of_continuation_in_ids + len(current_continuation_tokens_ns) > actual_sequence_len:
                 num_tokens_to_score = actual_sequence_len - start_of_continuation_in_ids
                 if num_tokens_to_score <= 0:
                    flat_choice_log_likelihoods.append(-float('inf'))
                    continue
            else:
                num_tokens_to_score = len(current_continuation_tokens_ns)

            log_probs_full_sequence = torch.nn.functional.log_softmax(current_logits_slice, dim=-1)
            current_choice_log_likelihood = 0.0
            valid_tokens_scored = 0

            for i in range(num_tokens_to_score):
                token_id_being_predicted = current_continuation_tokens_ns[i]
                idx_of_token_in_input_ids = start_of_continuation_in_ids + i

                if idx_of_token_in_input_ids == 0 or idx_of_token_in_input_ids >= actual_sequence_len:
                    if i == 0: current_choice_log_likelihood = -float('inf')
                    break

                log_prob_dist_for_token = log_probs_full_sequence[idx_of_token_in_input_ids - 1, :]
                current_choice_log_likelihood += log_prob_dist_for_token[token_id_being_predicted].item()
                valid_tokens_scored +=1

            if valid_tokens_scored > 0:
                flat_choice_log_likelihoods.append(current_choice_log_likelihood)
            else:
                flat_choice_log_likelihoods.append(-float('inf'))

        all_predicted_labels, all_probabilities, all_scores_grouped = [], [], []
        current_flat_idx = 0
        for doc_idx, num_choices in enumerate(doc_choice_counts):
            if num_choices == 0:
                all_predicted_labels.append("N/A"); all_probabilities.append([]); all_scores_grouped.append([])
                continue

            doc_scores = flat_choice_log_likelihoods[current_flat_idx : current_flat_idx + num_choices]
            all_scores_grouped.append(doc_scores)
            scores_array = np.asarray(doc_scores)

            # Softmax calculation, robust to -inf
            exp_scores = np.exp(scores_array - np.max(scores_array, initial=-np.inf))
            exp_scores[scores_array == -float('inf')] = 0
            sum_exp_scores = np.sum(exp_scores)
            probabilities = exp_scores / sum_exp_scores if sum_exp_scores > 0 else np.full(num_choices, 1.0 / num_choices)
            most_probable_choice_idx = np.argmax(probabilities) if probabilities.size > 0 else -1

            all_probabilities.append(probabilities.tolist())
            original_doc_choice_labels = self._get_choice_labels(batch_docs[doc_idx])
            pred_label = original_doc_choice_labels[most_probable_choice_idx] if 0 <= most_probable_choice_idx < len(original_doc_choice_labels) else "Error"
            all_predicted_labels.append(pred_label)
            current_flat_idx += num_choices

        return all_predicted_labels, all_probabilities, all_scores_grouped

    def calculate_accuracy(self, predicted_labels, ground_truth_labels):
        if len(predicted_labels) != len(ground_truth_labels):
            raise ValueError("Predicted and ground truth lists must have the same length.")
        if not predicted_labels: return 0.0
        correct = sum(1 for pred, truth in zip(predicted_labels, ground_truth_labels) if pred == truth)
        return correct / len(predicted_labels)

def format_batch(batch):
    """
    Transforms a batch from the Hugging Face dataset format to the
    list of dictionaries format expected by MCQProcessor.
    """
    formatted_docs = []
    num_items = len(batch['id'])
    for i in range(num_items):
        doc = {
            "id": batch['id'][i],
            "question": batch['question'][i],
            "choices": [
                batch['A'][i],
                batch['B'][i],
                batch['C'][i],
                batch['D'][i]
            ],
            "answer_label": batch['answer'][i]
        }
        formatted_docs.append(doc)
    return formatted_docs


def main():
    dataset_name = dataset_hf_path
    dataset_split = 'dev' ## This is to load the dev set from the dataset

    ## Select your batch size here (depending on the used LLM and the GPU Memory you have)
    batch_size = 4

    try:
        processor = MCQProcessor(model_name=model_name)
    except Exception:
        print("Failed to initialize MCQProcessor. Exiting.")
        return

    # --- Load Dataset ---
    print(f"\nLoading dataset '{dataset_name}' split '{dataset_split}'...")
    try:
        dev_data = load_dataset(dataset_name, split=dataset_split)
        print(f"Dataset loaded successfully with {len(dev_data)} examples.")
    except Exception as e:
        print(f"Failed to load dataset. Error: {e}")
        return

    # --- Batch Processing ---
    all_predictions = []
    all_ground_truths = []
    all_ids = []
    wrong_questions = []
    num_batches = (len(dev_data) + batch_size - 1) // batch_size

    print(f"\n--- Starting evaluation on {len(dev_data)} questions in {num_batches} batches ---")

    try:
        for i in range(0, len(dev_data), batch_size):
            print(f"Processing batch {i // batch_size + 1} / {num_batches}...")

            raw_batch = dev_data[i : i + batch_size]
            batch_docs = format_batch(raw_batch)

            batch_ids = [doc['id'] for doc in batch_docs]

            predicted_labels, all_probabilities, all_scores = processor.process_batch(batch_docs)
            ground_truth_labels = [doc["answer_label"] for doc in batch_docs]

            all_ids.extend(batch_ids)
            all_predictions.extend(predicted_labels)
            all_ground_truths.extend(ground_truth_labels)

            # Collect wrong predictions
            for doc_idx, (pred, truth, doc_id, doc) in enumerate(zip(predicted_labels, ground_truth_labels, batch_ids, batch_docs)):
                if pred != truth:
                    wrong_questions.append({
                        'id': doc_id,
                        'question': doc['question']
                    })

            if i == 0:
                print("\n--- Detailed Results for First Batch ---")
                for doc_idx, doc in enumerate(batch_docs):
                    print(f"\nDocument ID: {doc['id']}")
                    print(f"  Question: {doc['question']}")
                    doc_choice_prefixes = [p[0] for p in processor.choice_prefixes]
                    for j, choice_text in enumerate(doc["choices"]):
                        score = all_scores[doc_idx][j] if j < len(all_scores[doc_idx]) else 'N/A'
                        prob = all_probabilities[doc_idx][j] if j < len(all_probabilities[doc_idx]) else 'N/A'
                        print(f"    {doc_choice_prefixes[j]}. {choice_text} -> Score: {score:.4f}, Prob: {prob:.4f}")

                    pred = predicted_labels[doc_idx]
                    truth = ground_truth_labels[doc_idx]
                    print(f"  Predicted Label: '{pred}'")
                    print(f"  Ground Truth Label: '{truth}'")
                    print(f"  Result: {'CORRECT ✅' if pred == truth else 'INCORRECT ❌'}")
                print("\n--- (End of first batch details) ---\n")

        # --- Calculate and Display Final Accuracy ---
        print("\n--- Overall Evaluation Complete ---")
        accuracy = processor.calculate_accuracy(all_predictions, all_ground_truths)
        correct_count = sum(1 for p, t in zip(all_predictions, all_ground_truths) if p == t)
        total_count = len(all_ground_truths)

        print(f"Total Questions Evaluated: {total_count}")
        print(f"Correct Predictions: {correct_count}")
        print(f"Overall Accuracy: {accuracy:.2%}")

        print("\n--- Saving predictions to CSV ---")
        predictions_df = pd.DataFrame({
            'id': all_ids,
            'prediction': all_predictions
        })
        predictions_df.to_csv(f'/export/home/hhassan/palmx/predictions/{adapter_path.split("/")[1]}.csv', index=False)
        print(f"Predictions successfully saved to '{adapter_path.split('/')[1]}.csv'.")

        # Save wrong questions to file
        if wrong_questions:
            wrong_df = pd.DataFrame(wrong_questions)
            wrong_file_path = f'/export/home/hhassan/palmx/predictions/wrong_{adapter_path.split("/")[1]}.csv'
            wrong_df.to_csv(wrong_file_path, index=False)
            print(f"Wrong questions ({len(wrong_questions)}) saved to '{wrong_file_path}'.")
        else:
            print("No wrong predictions to save.")

    except Exception as e:
        print(f"\nAn error occurred during batch processing or evaluation: {e}")

if __name__ == "__main__":
    main()