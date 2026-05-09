import copy
import random

# typing 
from typing import List, Tuple
import time
import torch
import os
from datetime import datetime
from transformers import AutoTokenizer
# TODO
# from transformers import LlamaTokenizer
# tokenizer=LlamaTokenizer.from_pretrained("YOUR_MODEL_PATH")
# TOKENIZER = AutoTokenizer.from_pretrained("YOUR_MODEL_PATH")
TOPK = 10  # topk for sparse tree

# Debug logging configuration
DEBUG_REJECTION_SAMPLING = True  # Set to False to disable debug output
DEBUG_LOG_TO_FILE = True  # Set to False to only print to console
DEBUG_LOG_DIR = "mars_debug_logs"  # Directory to save debug logs

class DebugLogger:
    """Simple logger that outputs to both console and file"""
    def __init__(self):
        self.log_file = None
        if DEBUG_LOG_TO_FILE:
            # Create log directory if it doesn't exist
            os.makedirs(DEBUG_LOG_DIR, exist_ok=True)
            # Create log file with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"rejection_sampling_{timestamp}.log"
            log_path = os.path.join(DEBUG_LOG_DIR, log_filename)
            self.log_file = open(log_path, 'w', encoding='utf-8')
            print(f"[INFO] Debug log will be saved to: {log_path}")
    
    def log(self, message):
        """Print to console and optionally write to file"""
        if DEBUG_REJECTION_SAMPLING:
            print(message)
            if self.log_file is not None:
                self.log_file.write(message + '\n')
                self.log_file.flush()  # Ensure immediate write
    
    def close(self):
        """Close the log file"""
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None

class AcceptanceStats:
    """Track acceptance rate statistics for relaxation mechanism"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all statistics"""
        self.total_positions = 0  # Total positions evaluated
        self.relaxed_accept = 0   # Accepted via relaxation (top-k check)
        self.standard_accept = 0  # Accepted via standard rejection sampling
        self.rejected = 0         # Rejected tokens
        self.relaxation_triggered = 0  # Times relaxation was triggered
        self.standard_triggered = 0    # Times standard sampling was used
    
    def record_relaxed_accept(self):
        """Record a token accepted via relaxation"""
        self.total_positions += 1
        self.relaxed_accept += 1
        self.relaxation_triggered += 1
    
    def record_standard_accept(self):
        """Record a token accepted via standard rejection sampling"""
        self.total_positions += 1
        self.standard_accept += 1
        self.standard_triggered += 1
    
    def record_rejected(self, is_relaxation):
        """Record a rejected token"""
        self.total_positions += 1
        self.rejected += 1
        if is_relaxation:
            self.relaxation_triggered += 1
        else:
            self.standard_triggered += 1
    
    def get_stats(self):
        """Get statistics as a dictionary"""
        if self.total_positions == 0:
            return {
                'total_positions': 0,
                'acceptance_rate': 0.0,
                'relaxed_accept_rate': 0.0,
                'standard_accept_rate': 0.0,
                'rejection_rate': 0.0,
                'relaxation_usage_rate': 0.0,
            }
        
        return {
            'total_positions': self.total_positions,
            'relaxed_accept': self.relaxed_accept,
            'standard_accept': self.standard_accept,
            'rejected': self.rejected,
            'acceptance_rate': (self.relaxed_accept + self.standard_accept) / self.total_positions * 100,
            'relaxed_accept_rate': self.relaxed_accept / self.total_positions * 100,
            'standard_accept_rate': self.standard_accept / self.total_positions * 100,
            'rejection_rate': self.rejected / self.total_positions * 100,
            'relaxation_triggered': self.relaxation_triggered,
            'standard_triggered': self.standard_triggered,
            'relaxation_usage_rate': self.relaxation_triggered / self.total_positions * 100 if self.total_positions > 0 else 0.0,
        }
    
    def print_stats(self):
        """Print formatted statistics"""
        stats = self.get_stats()
        logger = get_debug_logger()
        logger.log("\n" + "="*80)
        logger.log("[ACCEPTANCE STATISTICS]")
        logger.log("="*80)
        logger.log(f"Total positions evaluated: {stats['total_positions']}")
        logger.log(f"")
        logger.log(f"Overall Acceptance Rate: {stats['acceptance_rate']:.2f}%")
        logger.log(f"  - Relaxed acceptance:  {stats['relaxed_accept']} ({stats['relaxed_accept_rate']:.2f}%)")
        logger.log(f"  - Standard acceptance: {stats['standard_accept']} ({stats['standard_accept_rate']:.2f}%)")
        logger.log(f"  - Rejected:            {stats['rejected']} ({stats['rejection_rate']:.2f}%)")
        logger.log(f"")
        logger.log(f"Relaxation mechanism triggered: {stats['relaxation_triggered']} times ({stats['relaxation_usage_rate']:.2f}%)")
        logger.log(f"Standard sampling used:         {stats['standard_triggered']} times ({100 - stats['relaxation_usage_rate']:.2f}%)")
        logger.log("="*80 + "\n")
        
        return stats

class LogitsCollector:
    """Collect top-2 logits distribution data for visualization"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all collected data"""
        self.all_top2_logits = []  # List of (logit_1st, logit_2nd) tuples
        self.all_top2_probs = []   # List of (prob_1st, prob_2nd) tuples
        self.relaxed_top2_logits = []  # Logits where relaxation was triggered
        self.relaxed_top2_probs = []   # Probs where relaxation was triggered
        self.accepted_top2_logits = []  # Logits where relaxation actually accepted
        self.accepted_top2_probs = []   # Probs where relaxation actually accepted
        self.logit_ratios = []  # All computed logit ratios
        self.prob_ratios = []   # All computed prob ratios
        self._pending_record = None  # Temporary storage for pending record
    
    def record(self, logit_1st, logit_2nd, prob_1st, prob_2nd, use_relaxation):
        """Record top-2 logits and probabilities (first step)"""
        self.all_top2_logits.append((logit_1st, logit_2nd))
        self.all_top2_probs.append((prob_1st, prob_2nd))
        
        # Compute ratios
        logit_ratio = logit_2nd / (logit_1st + 1e-10)
        prob_ratio = prob_2nd / (prob_1st + 1e-10)
        self.logit_ratios.append(logit_ratio)
        self.prob_ratios.append(prob_ratio)
        
        if use_relaxation:
            self.relaxed_top2_logits.append((logit_1st, logit_2nd))
            self.relaxed_top2_probs.append((prob_1st, prob_2nd))
            # Store pending record for potential acceptance update
            self._pending_record = (logit_1st, logit_2nd, prob_1st, prob_2nd)
        else:
            self._pending_record = None
    
    def record_accepted(self):
        """Call this when a relaxation-triggered token is actually accepted"""
        if self._pending_record is not None:
            logit_1st, logit_2nd, prob_1st, prob_2nd = self._pending_record
            self.accepted_top2_logits.append((logit_1st, logit_2nd))
            self.accepted_top2_probs.append((prob_1st, prob_2nd))
            self._pending_record = None
    
    def clear_pending(self):
        """Clear pending record (called when relaxation triggered but not accepted)"""
        self._pending_record = None
    
    def save_to_file(self, filepath="logits_distribution.json"):
        """Save collected data to JSON file"""
        import json
        data = {
            "all_top2_logits": self.all_top2_logits,
            "all_top2_probs": self.all_top2_probs,
            "relaxed_top2_logits": self.relaxed_top2_logits,
            "relaxed_top2_probs": self.relaxed_top2_probs,
            "accepted_top2_logits": self.accepted_top2_logits,
            "accepted_top2_probs": self.accepted_top2_probs,
            "logit_ratios": self.logit_ratios,
            "prob_ratios": self.prob_ratios,
            "summary": {
                "total_samples": len(self.all_top2_logits),
                "relaxation_triggered": len(self.relaxed_top2_logits),
                "relaxation_accepted": len(self.accepted_top2_logits),
                "relaxation_trigger_rate": len(self.relaxed_top2_logits) / max(1, len(self.all_top2_logits)) * 100,
                "relaxation_accept_rate": len(self.accepted_top2_logits) / max(1, len(self.relaxed_top2_logits)) * 100 if self.relaxed_top2_logits else 0
            }
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[INFO] Logits distribution saved to: {filepath}")
        return filepath
    
    def get_stats(self):
        """Get summary statistics"""
        import numpy as np
        if len(self.all_top2_logits) == 0:
            return {"total_samples": 0}
        
        logits_1st = [x[0] for x in self.all_top2_logits]
        logits_2nd = [x[1] for x in self.all_top2_logits]
        
        return {
            "total_samples": len(self.all_top2_logits),
            "relaxation_triggered": len(self.relaxed_top2_logits),
            "relaxation_accepted": len(self.accepted_top2_logits),
            "logit_1st_range": (min(logits_1st), max(logits_1st)),
            "logit_2nd_range": (min(logits_2nd), max(logits_2nd)),
            "logit_ratio_range": (min(self.logit_ratios), max(self.logit_ratios)),
            "prob_ratio_range": (min(self.prob_ratios), max(self.prob_ratios)),
            "negative_logit_1st_count": sum(1 for x in logits_1st if x < 0),
            "negative_logit_2nd_count": sum(1 for x in logits_2nd if x < 0),
        }

# Global logits collector instance
_logits_collector = None

def get_logits_collector():
    """Get or create the global logits collector"""
    global _logits_collector
    if _logits_collector is None:
        _logits_collector = LogitsCollector()
    return _logits_collector

def reset_logits_collector():
    """Reset logits collector"""
    collector = get_logits_collector()
    collector.reset()

def save_logits_distribution(filepath="logits_distribution.json"):
    """Save collected logits distribution to file"""
    collector = get_logits_collector()
    return collector.save_to_file(filepath)

# Global acceptance statistics instance
_acceptance_stats = None

def get_acceptance_stats():
    """Get or create the global acceptance statistics tracker"""
    global _acceptance_stats
    if _acceptance_stats is None:
        _acceptance_stats = AcceptanceStats()
    return _acceptance_stats

def reset_acceptance_stats():
    """Reset acceptance statistics"""
    stats = get_acceptance_stats()
    stats.reset()

def print_acceptance_stats():
    """Print current acceptance statistics"""
    stats = get_acceptance_stats()
    return stats.print_stats()

# Global debug logger instance
_debug_logger = None

def get_debug_logger():
    """Get or create the global debug logger"""
    global _debug_logger
    if _debug_logger is None:
        _debug_logger = DebugLogger()
    return _debug_logger

from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)


class Timer:
    def __init__(self,name):
        self.name = name
    def __enter__(self):
        torch.cuda.synchronize()
        self.start = time.perf_counter()


    def __exit__(self, exc_type, exc_value, traceback):
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.start
        print(f'{self.name} took {elapsed} seconds')


def prepare_logits_processor(
        temperature: float = 0.0,
        repetition_penalty: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 0
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    if temperature > 1e-5:
        if temperature >= 1e-5 and temperature != 1.0:
            processor_list.append(TemperatureLogitsWarper(temperature))
        if repetition_penalty > 1.0:
            processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
        if 1e-8 <= top_p < 1.0:
            processor_list.append(TopPLogitsWarper(top_p))
        if top_k > 0:
            processor_list.append(TopKLogitsWarper(top_k))
    return processor_list


# test_processor = prepare_logits_processor(
#         0.0, 0.0, -1, 1
#     )


def pad_path(path: List[int], length: int, pad_value: int = -2) -> List[int]:
    """
    Pad the given path list with a specific value up to a specified length.

    Parameters:
    - path (list): The original list that needs padding.
    - length (int): The desired length of the padded list.
    - pad_value (optional, default=-2): The value to use for padding.

    Returns:
    - list: A new list based on the original path but padded to the desired length.

    Example:
    >>> pad_path([1,2,3], 5)
    [1, 2, 3, -2, -2]

    Note:
    If the given path is already longer than the specified length,
    then no padding occurs, and the original path is returned.
    """

    # Calculate the number of padding values needed by subtracting the length
    # of the path from the desired length.
    # Append the padding values to the original path and return the new list.
    return path + [pad_value] * (length - len(path))


def generate_tree_buffers(tree_choices, device="cuda"):
    def custom_sort(lst):
        # sort_keys=[len(list)]
        sort_keys = []
        for i in range(len(lst)):
            sort_keys.append(lst[i] if lst[i] >= 0 else maxitem)
        return sort_keys
    with Timer("sort"):

        sorted_tree_choices = sorted(tree_choices, key=lambda x: (len(x), x))
        tree_len = len(sorted_tree_choices) + 1

    # Initialize depth_counts to keep track of how many choices have a particular depth
        depth_counts = []
        prev_depth = 0
        for path in sorted_tree_choices:
            depth = len(path)
            if depth != prev_depth:
                depth_counts.append(0)
            depth_counts[depth - 1] += 1
            prev_depth = depth

        tree_attn_mask = torch.eye(tree_len, tree_len)
        tree_attn_mask[:, 0] = 1
        start = 0
        for i in range(len(depth_counts)):
            for j in range(depth_counts[i]):
                cur_tree_choice = sorted_tree_choices[start + j]
                # retrieve ancestor position
                if len(cur_tree_choice) == 1:
                    continue
                ancestor_idx = []
                for c in range(len(cur_tree_choice) - 1):
                    ancestor_idx.append(sorted_tree_choices.index(cur_tree_choice[:c + 1]) + 1)
                tree_attn_mask[j + start + 1, ancestor_idx] = 1
            start += depth_counts[i]

        tree_indices = torch.zeros(tree_len, dtype=torch.long)
        p_indices = [0 for _ in range(tree_len - 1)]
        b_indices = [[] for _ in range(tree_len - 1)]
        tree_indices[0] = 0
        start = 0
        bias = 0
        for i in range(len(depth_counts)):
            inlayer_bias = 0
            b = []
            for j in range(depth_counts[i]):
                cur_tree_choice = sorted_tree_choices[start + j]
                cur_parent = cur_tree_choice[:-1]
                if j != 0:
                    if cur_parent != parent:
                        bias += 1
                        inlayer_bias += 1
                        parent = cur_parent
                        b = []
                else:
                    parent = cur_parent
                tree_indices[start + j + 1] = cur_tree_choice[-1] + TOPK * (i + bias) + 1
                p_indices[start + j] = inlayer_bias
                if len(b) > 0:
                    b_indices[start + j] = copy.deepcopy(b)
                else:
                    b_indices[start + j] = []
                b.append(cur_tree_choice[-1] + TOPK * (i + bias) + 1)
            start += depth_counts[i]

        p_indices = [-1] + p_indices
        tree_position_ids = torch.zeros(tree_len, dtype=torch.long)
        start = 0
        for i in range(len(depth_counts)):
            tree_position_ids[start + 1: start + depth_counts[i] + 1] = i + 1
            start += depth_counts[i]

        retrieve_indices_nest = []
        retrieve_paths = []
        for i in range(len(sorted_tree_choices)):
            cur_tree_choice = sorted_tree_choices[-i - 1]
            retrieve_indice = []
            if cur_tree_choice in retrieve_paths:
                continue
            else:
                for c in range(len(cur_tree_choice)):
                    retrieve_indice.append(sorted_tree_choices.index(cur_tree_choice[:c + 1]))
                    retrieve_paths.append(cur_tree_choice[:c + 1])
            retrieve_indices_nest.append(retrieve_indice)
        max_length = max([len(x) for x in retrieve_indices_nest])
        retrieve_indices = [pad_path(path, max_length) for path in retrieve_indices_nest]
        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
        retrieve_indices = retrieve_indices + 1
        retrieve_indices = torch.cat([torch.zeros((retrieve_indices.shape[0], 1), dtype=torch.long), retrieve_indices],
                                     dim=1)

        maxitem = retrieve_indices.max().item() + 5



        retrieve_indices = retrieve_indices.tolist()
        retrieve_indices = sorted(retrieve_indices, key=custom_sort)
        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)



    # Aggregate the generated buffers into a dictionary
    tree_buffers = {
        "tree_attn_mask": tree_attn_mask.unsqueeze(0).unsqueeze(0),
        "tree_indices": tree_indices,
        "tree_position_ids": tree_position_ids,
        "retrieve_indices": retrieve_indices,
    }

    # Move the tensors in the dictionary to the specified device
    tree_buffers = {
        k: v.clone().to(device)
        if isinstance(v, torch.Tensor)
        else torch.tensor(v, device=device)
        for k, v in tree_buffers.items()
    }

    return tree_buffers


def initialize_tree0(input_ids, model, past_key_values, logits_processor):
    draft_tokens, retrieve_indices,tree_mask,tree_position_ids, outputs, logits, hidden_state, sample_token = model(
        input_ids, past_key_values=past_key_values, output_orig=True, logits_processor=logits_processor
    )

    #     if logits_processor is not None:
    #         logits = orig[:, -1]
    #         logits = logits_processor(None, logits)
    #         probabilities = torch.nn.functional.softmax(logits, dim=1)
    #         token = torch.multinomial(probabilities, 1)
    #     else:
    #         token = torch.argmax(orig[:, -1])
    #         token = token[None, None]
    #     input_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1)
    #     # Clone the output hidden states
    #
    #     draft_tokens, retrieve_indices,tree_mask,tree_position_ids = self.ea_layer.topK_genrate(hidden_states, input_ids, self.base_model.lm_head)
    #     if output_orig:
    #         return draft_tokens, retrieve_indices,tree_mask,tree_position_ids, outputs, orig, hidden_states, token
    #     return draft_tokens, retrieve_indices,tree_mask,tree_position_ids, hidden_states, token
    return draft_tokens, retrieve_indices,tree_mask,tree_position_ids, logits, hidden_state, sample_token

def initialize_tree(input_ids, model, past_key_values, logits_processor):
    outputs, orig, hidden_states = model(
        input_ids, past_key_values=past_key_values, output_orig=True
    )

    if logits_processor is not None:
        logits = orig[:, -1]
        logits = logits_processor(None, logits)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        token = torch.multinomial(probabilities, 1)
    else:
        token = torch.argmax(orig[:, -1])
        token = token[None, None]
    input_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1)

    # Clone the output hidden states
    if model.use_mars:
        ea_device = model.ea_layer.lm_head.weight.device
        if outputs["hidden_states"][0].device != ea_device:
            outputs["hidden_states"] = [x.to(ea_device) for x in outputs["hidden_states"]]
        hidden_states=torch.cat(outputs["hidden_states"],dim=-1)
    draft_tokens, retrieve_indices,tree_mask,tree_position_ids, retrieve_probs = model.ea_layer.topK_genrate(hidden_states, input_ids, model.base_model.lm_head,logits_processor)
    return draft_tokens, retrieve_indices,tree_mask,tree_position_ids, orig, hidden_states, token, retrieve_probs


def reset_tree_mode(
        model,
):
    model.base_model.model.tree_mask = None
    model.base_model.model.tree_mode = None


def reset_past_key_values(passed_key_values: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Resets the current lengths in the passed key-values to zero.

    This function is designed to be used during the evaluation of a baseline model.
    It iterates through each layer's key-values and sets their current lengths to zero,
    effectively resetting their state.

    Args:
    - passed_key_values (list of torch.Tensor): Contains past hidden states and past attention values for each layer.

    Returns:
    - passed_key_values (list of torch.Tensor): Updated past hidden states and past attention values with reset lengths.
    """
    for i in range(len(passed_key_values)):
        for j in range(2):
            passed_key_values[i][j].current_length.fill_(0)
    return passed_key_values


def generate_candidates(tree_logits, tree_indices, retrieve_indices, sample_token, logits_processor, retrieve_probs=None):
    sample_token = sample_token.to(tree_indices.device)

    candidates_logit = sample_token[0]

    candidates_tree_logits = tree_logits

    candidates = torch.cat([candidates_logit, candidates_tree_logits.view(-1)], dim=-1)

    tree_candidates = candidates[tree_indices]

    tree_candidates_ext = torch.cat(
        [tree_candidates, torch.zeros((1), dtype=torch.long, device=tree_candidates.device) - 1], dim=0)

    cart_candidates = tree_candidates_ext[retrieve_indices]


    # Unsqueeze the tree candidates for dimension consistency.
    tree_candidates = tree_candidates.unsqueeze(0)
    return cart_candidates,  tree_candidates, retrieve_probs


def tree_decoding(
        model,
        tree_candidates,
        past_key_values,
        tree_position_ids,
        input_ids,
        retrieve_indices,
):
    position_ids = tree_position_ids + input_ids.shape[1]
    if position_ids is not None and position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)
    outputs, tree_logits, hidden_state = model(
        tree_candidates,
        output_orig=True,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )

    if model.use_mars:
        ea_device = model.ea_layer.lm_head.weight.device
        if outputs["hidden_states"][0].device != ea_device:
            outputs["hidden_states"] = [x.to(ea_device) for x in outputs["hidden_states"]]
        hidden_state = torch.cat(outputs["hidden_states"], dim=-1)

    logits = tree_logits[0, retrieve_indices]
    return logits, hidden_state, outputs


def evaluate_posterior(
        logits: torch.Tensor,
        candidates: torch.Tensor,
        logits_processor,
        retrieve_probs=None,
        relaxation_threshold=0.9,
):
    """
    Evaluate the posterior probabilities of the candidates based on the provided logits and choose the best candidate.

    Depending on the temperature value, the function either uses greedy decoding or evaluates posterior
    probabilities to select the best candidate.

    Args:
    - logits (torch.Tensor): Predicted logits of shape (batch_size, sequence_length, vocab_size).
    - candidates (torch.Tensor): Candidate token sequences.
    - temperature (float): Softmax temperature for probability scaling. A value of 0 indicates greedy decoding.
    - posterior_threshold (float): Threshold for posterior probability.
    - posterior_alpha (float): Scaling factor for the threshold.
    - retrieve_probs (torch.Tensor): Draft model probabilities for each candidate token.
    - relaxation_threshold (float): Threshold for top-2 logits ratio (default=0.87).
                                    When ratio > threshold, use relaxed acceptance (top-k check).

    Returns:
    - best_candidate (torch.Tensor): Index of the chosen best candidate.
    - accept_length (int): Length of the accepted candidate sequence.
    """
    # Greedy decoding based on temperature value
    if logits_processor is None:
        # Find the tokens that match the maximum logits for each position in the sequence
        posterior_mask = (
                candidates[:, 1:].to(logits.device) == torch.argmax(logits[:, :-1], dim=-1)
        ).int()
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        accept_length = candidates_accept_length.max()
        # Choose the best candidate
        if accept_length == 0:
            # Default to the first candidate if none are accepted
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
        else:
            best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
        return best_candidate, accept_length, logits[best_candidate, accept_length]

    else:
        
        accept_length = 1
        accept_cand = candidates[0][:1]
        best_candidate = 0
        for i in range(1, candidates.shape[1]):
            if i != accept_length:
                break
            adjustflag = False
            is_eq = (candidates[:, :accept_length] == accept_cand).all(dim=1)
            fi = torch.nonzero(is_eq, as_tuple=True)[0][0]
            gt_logits = logits[fi, i - 1][None]
            gt_logits = logits_processor(None, gt_logits)[0]
            gtp = torch.softmax(gt_logits, dim=0)
            
            # Calculate top-2 logits ratio for relaxation mechanism
            top2_logits, top2_indices = torch.topk(gt_logits, k=min(2, gt_logits.shape[0]))
            if top2_logits.shape[0] >= 2:
                # ratio = logit_2nd / logit_1st
                logit_ratio = top2_logits[1].item() / (top2_logits[0].item() + 1e-10)
                use_relaxation = logit_ratio > relaxation_threshold
                
                top2_tokens = top2_indices.tolist()
                
                # Collect logits data for visualization
                top2_probs = gtp[top2_indices]
                collector = get_logits_collector()
                collector.record(
                    logit_1st=top2_logits[0].item(),
                    logit_2nd=top2_logits[1].item(),
                    prob_1st=top2_probs[0].item(),
                    prob_2nd=top2_probs[1].item(),
                    use_relaxation=use_relaxation
                )
            else:
                logit_ratio = 0.0
                # logit_diff = float('inf')
                use_relaxation = False
                top2_tokens = top2_indices.tolist()
            
            candidates_set = []

            for j in range(candidates.shape[0]):
                if is_eq[j]:
                    x = candidates[j, i]
                    xi = x.item()
                    if xi in candidates_set or xi == -1:
                        continue
                    candidates_set.append(xi)

                    # Check if we should use relaxation mechanism
                    if use_relaxation:
                        # Relaxed acceptance: check if draft token is in top-k (top-2)
                        if xi in top2_tokens:
                            # Record that relaxation actually accepted this token
                            collector.record_accepted()
                            accept_cand = torch.cat((accept_cand, x[None]), dim=0)
                            accept_length += 1
                            best_candidate = j
                            break
                        else:
                            # Relaxation triggered but draft token not in top-2, clear pending
                            collector.clear_pending()
                            continue
                    else:
                        # Standard rejection sampling
                        r = random.random()
                        px = gtp[xi].item()
                        # Use actual draft model probability instead of hardcoded 1.0
                        if retrieve_probs is not None:
                            qx = retrieve_probs[j, i].item()
                            # Ensure qx is not zero to avoid division by zero
                            qx = max(qx, 1e-10)
                        else:
                            qx = 1.0
                        acp = min(1.0, px / qx)
                        
                        if r <= acp:
                            accept_cand = torch.cat((accept_cand, x[None]), dim=0)
                            accept_length += 1
                            best_candidate = j
                            break
                        else:
                            adjusted_prob = max(0.0, px - qx)
                            gtp[xi] = adjusted_prob
                            gtp = gtp / gtp.sum()
                            adjustflag = True
        if adjustflag and accept_length != candidates.shape[1]:
            sample_p = gtp
        else:
            gt_logits = logits[best_candidate, accept_length - 1][None]
            gt_logits = logits_processor(None, gt_logits)[0]
            sample_p = torch.softmax(gt_logits, dim=0)
        
        return torch.tensor(best_candidate), accept_length - 1, sample_p

# def evaluate_posterior(
#         logits: torch.Tensor,
#         candidates: torch.Tensor,
#         logits_processor,
#         retrieve_probs=None,
# ):
#     """
#     Evaluate posterior probabilities of the candidates using standard rejection sampling
#     (SD algorithm) and select the best candidate.

#     Args:
#         logits (torch.Tensor):
#             Target model logits of shape (batch_size, seq_len, vocab_size).
#         candidates (torch.Tensor):
#             Candidate token sequences of shape (num_candidates, seq_len).
#         logits_processor:
#             A logits processor (e.g. transformers.LogitsProcessorList) applied to
#             the target model logits before softmax.
#         retrieve_probs (torch.Tensor, optional):
#             Draft model probabilities for each candidate token, shape
#             (num_candidates, seq_len). Used as q(x) in rejection sampling.
#             If None, q(x) is treated as 1.0 for all tokens.
#         relaxation_threshold (float):
#             Unused in this implementation; kept for backward compatibility.
            
#     """

#     # Case 1: No logits_processor → greedy matching path
#     if logits_processor is None:
#         posterior_mask = (
#             candidates[:, 1:].to(logits.device) == torch.argmax(logits[:, :-1], dim=-1)
#         ).int()

#         candidates_accept_length = torch.cumprod(posterior_mask, dim=1).sum(dim=1)
#         accept_length = candidates_accept_length.max()

#         if accept_length == 0:
#             best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
#         else:
#             best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
#         # Return logits at the position right after the accepted prefix
#         return best_candidate, int(accept_length), logits[best_candidate, accept_length]

#     # Case 2: logits_processor is provided → standard SD rejection sampling
#     accept_length = 1
#     accept_cand = candidates[0][:1]  # current accepted prefix (tokens)
#     best_candidate = 0

#     # Flag indicating whether we have ever adjusted the distribution gtp during rejection
#     adjustflag = False

#     # Iterate over positions in the sequence
#     for i in range(1, candidates.shape[1]):
#         # We only accept tokens at the current accept_length position.
#         # If i diverges from accept_length, we stop extending.
#         if i != accept_length:
#             break

#         # Find all candidates whose prefix matches the currently accepted prefix
#         is_eq = (candidates[:, :accept_length] == accept_cand).all(dim=1)
#         fi = torch.nonzero(is_eq, as_tuple=True)[0][0]

#         # Get target logits at the previous position for one matching candidate
#         gt_logits = logits[fi, i - 1][None]
#         gt_logits = logits_processor(None, gt_logits)[0]
#         gtp = torch.softmax(gt_logits, dim=0)  # p(x) distribution over vocab

#         candidates_set = []

#         # Loop over all candidates that share the same prefix
#         for j in range(candidates.shape[0]):
#             if not is_eq[j]:
#                 continue

#             x = candidates[j, i]
#             xi = x.item()

#             # Skip duplicate tokens and invalid tokens (e.g. -1)
#             if xi in candidates_set or xi == -1:
#                 continue
#             candidates_set.append(xi)

#             # Standard rejection sampling:
#             #   accept with probability a(x) = min(1, p(x) / q(x))
#             r = random.random()
#             px = gtp[xi].item()

#             # Draft model probability q(x)
#             if retrieve_probs is not None:
#                 qx = retrieve_probs[j, i].item()
#                 qx = max(qx, 1e-10)  # avoid division by zero
#             else:
#                 qx = 1.0

#             acp = min(1.0, px / qx)

#             if r <= acp:
#                 # Accept this token, extend the accepted prefix
#                 accept_cand = torch.cat((accept_cand, x[None]), dim=0)
#                 accept_length += 1
#                 best_candidate = j
#                 break
#             else:
#                 # Rejection: update the distribution
#                 # p'(x) = max(0, p(x) - q(x)), then renormalize
#                 adjusted_prob = max(0.0, px - qx)
#                 gtp[xi] = adjusted_prob
#                 gtp = gtp / gtp.sum()
#                 adjustflag = True

#     # After the loop, decide which distribution to return as sample_p
#     if adjustflag and accept_length != candidates.shape[1]:
#         # If we adjusted the distribution and did not consume the whole sequence,
#         # use the adjusted distribution gtp.
#         sample_p = gtp
#     else:
#         # Otherwise, recompute logits for the last accepted position of the
#         # chosen candidate and take softmax.
#         gt_logits = logits[best_candidate, accept_length - 1][None]
#         gt_logits = logits_processor(None, gt_logits)[0]
#         sample_p = torch.softmax(gt_logits, dim=0)

#     return (
#         torch.tensor(best_candidate, device=candidates.device, dtype=torch.long),
#         accept_length - 1,
#         sample_p,
#     )


@torch.no_grad()
def update_inference_inputs(
        input_ids,
        candidates,
        best_candidate,
        accept_length,
        retrieve_indices,
        logits_processor,
        new_token,
        past_key_values_data_list,
        current_length_data,
        model,
        hidden_state_new,
        sample_p
):
    prev_input_len = input_ids.shape[1]
    # Map the best candidate indices to the original indices in the sequence
    select_indices = (
            retrieve_indices[best_candidate, : accept_length + 1] + prev_input_len
    )
    # Append the tokens from the best candidate to the input sequence
    input_ids = torch.cat(
        [input_ids, candidates[None, best_candidate, : accept_length + 1].to(input_ids.device)], dim=-1
    )
    # Update the past key values based on the selected tokens
    # Source tensor that contains relevant past information based on the selected candidate
    for past_key_values_data in past_key_values_data_list:
        tgt = past_key_values_data[..., select_indices.to(past_key_values_data.device), :]
        # Destination tensor where the relevant past information will be stored
        dst = past_key_values_data[..., prev_input_len: prev_input_len + tgt.shape[-2], :]
        # Copy relevant past information from the source to the destination
        dst.copy_(tgt, non_blocking=True)

    # Update the current length tensor (currently only support batch size is 1)
    current_length_data.fill_(prev_input_len + tgt.shape[-2])

    retrieve_hidden_state_new = hidden_state_new[:, retrieve_indices]
    accept_hidden_state_new = retrieve_hidden_state_new[:, best_candidate, : accept_length + 1]
    # token=model.base_model.lm_head(accept_hidden_state_new[:,-1]).argmax()
    # token=token[None,None]
    prob = sample_p
    if logits_processor is not None:
        token = torch.multinomial(prob, 1)
        token = token[None]
    else:
        token = torch.argmax(prob)
        token = token[None, None]
    # hidden_state = torch.cat((hidden_state, accept_hidden_state_new), dim=1)
    # logger = get_debug_logger()
    # logger.log(f"it should be {token}")
    # logger.log(f"accept text:{TOKENIZER.decode(token.squeeze().tolist())}")
    draft_tokens, retrieve_indices,tree_mask,tree_position_ids, retrieve_probs = model.ea_layer.topK_genrate(accept_hidden_state_new,
                                              input_ids=torch.cat((input_ids, token.to(input_ids.device)), dim=1),
                                              head=model.base_model.lm_head,logits_processor=logits_processor)


    new_token += accept_length + 1

    return input_ids, draft_tokens, retrieve_indices,tree_mask,tree_position_ids, new_token, None, token, retrieve_probs


if __name__ == "__main__":
    logits = torch.randn(1, 5)
    tp = prepare_logits_processor(0.9, 0, 0.9, 0)
    l = tp(None, logits)
    if tp is None:
        print(tp)
