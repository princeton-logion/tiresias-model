#!/usr/bin/env python3
"""
Character-level, span-based dynamic masking collator w/ multiple scheduling strategies
"""

import torch
import numpy as np
import random
import math
from typing import List, Dict, Tuple, Optional, Iterable
from enum import Enum


class MaskingScheduleType(Enum):
    CONSTANT = "constant"
    LINEAR = "linear"
    COSINE = "cosine"


class DataCollatorForMaskedLM:
    """
    Data collator with:
        - masking schedule
        - Poisson-distributed spans
        - BERT-style masking
        - excluded tokens list
    """
    
    def __init__(self,
                 mask_token_id: int,
                 pad_token_id: int,
                 vocab_size: int,
                 seed: Optional[int] = None,
                 mlm_probability: float = 0.15,
                 lambda_param: float = 6.0,
                 max_span_length: int = 10,
                 mask_replace_prob: float = 0.8,
                 random_replace_prob: float = 0.1,
                 excluded_token_ids: Optional[Iterable[int]] = (0, 1, 2, 3, 4, 5),
                 mlm_schedule_type: str = "linear",
                 initial_mlm_prob: Optional[float] = .40,
                 final_mlm_prob: Optional[float] = .15,
                 total_training_steps: Optional[int] = 182000,
                 num_cycles: float = 0.5):
        """
        Initialize dynamic masking collator
        
        Parameters:
            mask_token_id: MASK] token id
            pad_token_id: [PAD] token id
            vocab_size: vocabulary size
            seed: optional seed
            mlm_probability: constant schedule MLM probability
            lambda_param: mean span length of Poisson distribution
            max_span_length: maximum span length
            mask_replace_prob: percentage of positions replaced with [MASK]
            random_replace_prob: percentage of positions replaced with random token
            excluded_token_ids: token IDs excluded from masking and random replacement
            mlm_schedule_type: schedule type "constant", "linear", "cosine"
            initial_mlm_prob: start masking rate for linear/cosine mask schedules
            final_mlm_prob: end masking rate for linear/cosine mask schedules
            total_training_steps: total number of training steps
            num_cycles: Nunumber of cosine cycles for cosine schedule
        """
        self.seed = seed
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.vocab_size = vocab_size
        self.mlm_probability = mlm_probability
        self.lambda_param = lambda_param
        self.max_span_length = max_span_length

        self.mask_replace_prob = mask_replace_prob
        self.random_replace_prob = random_replace_prob
        if not 0.0 <= mask_replace_prob <= 1.0 or not 0.0 <= random_replace_prob <= 1.0:
            raise ValueError("mask_replace_prob and random_replace_prob must each be in [0, 1]")
        if mask_replace_prob + random_replace_prob > 1.0:
            raise ValueError("mask_replace_prob + random_replace_prob must not exceed 1")

        # ids excluded from replacement pool and as masking+replacement targets
        self._excluded_token_ids = set() if excluded_token_ids is None else set(excluded_token_ids)
        self._random_replacement_ids = [i for i in range(vocab_size) if i not in self._excluded_token_ids]
        if not self._random_replacement_ids:
            raise ValueError("excluded_token_ids leaves no token IDs available for random replacement")
        
        self.mlm_schedule_type = MaskingScheduleType(mlm_schedule_type)
        self.initial_mlm_prob = initial_mlm_prob if initial_mlm_prob is not None else mlm_probability
        self.final_mlm_prob = final_mlm_prob if final_mlm_prob is not None else mlm_probability
        self.total_training_steps = total_training_steps
        self.num_cycles = num_cycles
        
        self.current_step = 0
        
        self._validate_schedule_params()
        
        self._py_rng: Optional[random.Random] = None
        self._np_rng = None
        self._rngs_ready = False
    
    def _ensure_rngs(self):
        """
        create collator's private RNGs
        """
        if self._rngs_ready:
            return
        if self.seed is None:
            base_seed = None
        else:
            worker_info = torch.utils.data.get_worker_info()
            base_seed = self.seed if worker_info is None else worker_info.seed
        self._py_rng = random.Random(base_seed)
        self._np_rng = np.random.default_rng(base_seed)
        self._rngs_ready = True

    def _validate_schedule_params(self):
        """
        Validate scheduling parameters
        """
        if self.mlm_schedule_type != MaskingScheduleType.CONSTANT:
            if self.total_training_steps is None:
                raise ValueError(f"total_training_steps must be provided for {self.mlm_schedule_type.value} scheduling")
            
            print(f"Initialized {self.mlm_schedule_type.value} masking schedule: {self.initial_mlm_prob:.2f} → {self.final_mlm_prob:.2f} over {self.total_training_steps} steps")
    
    def get_current_mlm_prob(self) -> float:
        """
        Calculate current masking prob
        
        Returns:
            Current masking prob
        """
        if self.mlm_schedule_type == MaskingScheduleType.CONSTANT:
            return self.mlm_probability
        
        if self.total_training_steps is None or self.total_training_steps == 0:
            return self.mlm_probability
        
        progress = min(self.current_step / self.total_training_steps, 1.0)
        
        if self.mlm_schedule_type == MaskingScheduleType.LINEAR:
            return self._linear_schedule(progress)
        elif self.mlm_schedule_type == MaskingScheduleType.COSINE:
            return self._cosine_schedule(progress)
        else:
            raise ValueError(f"Unknown schedule type: {self.mlm_schedule_type}")
    
    def _linear_schedule(self, progress: float) -> float:
        """
        Linear masking schedule
        
        Parameters:
            progress (float) -- training progress (0.0-1.0)
        """
        return self.initial_mlm_prob + progress * (self.final_mlm_prob - self.initial_mlm_prob)
    
    def _cosine_schedule(self, progress: float) -> float:
        """
        Cosine masking schedule
        
        Parameters:
            progress (float) -- training progress (0.0-1.0)
        """
        cosine_coeff = 0.5 * (1.0 + math.cos(math.pi * self.num_cycles * 2.0 * progress))
        
        return self.final_mlm_prob + (self.initial_mlm_prob - self.final_mlm_prob) * cosine_coeff
    
    def set_step(self, step: int):
        """
        Update training step per schedule
        
        Parameters:
            step: current training step number
        """
        self.current_step = step
    
    def get_schedule_info(self) -> Dict[str, float]:
        """
        Get mask schedule info
        
        Returns:
            Dict with schedule info
        """
        current_prob = self.get_current_mlm_prob()
        progress = self.current_step / self.total_training_steps if self.total_training_steps else 0
        
        info = {
            'current_step': self.current_step,
            'total_steps': self.total_training_steps,
            'current_mlm_prob': current_prob,
            'progress': progress,
            'mlm_schedule_type': self.mlm_schedule_type.value
        }
        
        return info
    
    def _sample_poisson_span_length(self, lambda_param: float, max_length: int = 10) -> int:
        """
        Span determined by Poisson dist <= max_length
        
        Parameters:
            lambda_param (float) -- Poisson dist (λ) mean
            max_length (int) -- max span length in characters
        
        Returns:
            span length >1 < max_length (inclusive)
        """
        length = self._np_rng.poisson(lambda_param)
        
        length = max(1, length)
        
        return min(length, max_length)
    
    def _select_mask_spans(self, seq_length: int, current_mlm_prob: float,
                           excluded_positions: Optional[set] = None) -> List[Tuple[int, int]]:
        """
        Select spans to mask according to Poisson dist w/ dynamic masking
        
        Parameters:
            seq_length (int) -- sequence length
            current_mlm_prob (float) -- masking probability
            excluded_positions (set) -- positions to never mask
            
        Returns:
            Sorted list of (start, end) spans
        """
        if seq_length < 1:
            return []
        
        excluded = {p for p in excluded_positions if 0 <= p < seq_length} if excluded_positions else set()

        effective_length = max(0, seq_length - len(excluded))
        target_masked_chars = max(1, int(effective_length * current_mlm_prob))
        
        spans = []
        masked_positions = set()
        buffer_positions = set(excluded)
        attempts = 0
        max_attempts = effective_length * 2
        
        while len(masked_positions) < target_masked_chars and attempts < max_attempts:
            attempts += 1
            
            # skip buffers
            available_positions = [i for i in range(seq_length) 
                                 if i not in masked_positions and i not in buffer_positions]
            if not available_positions:
                break
            
            start = self._py_rng.choice(available_positions)
            
            span_length = self._sample_poisson_span_length(self.lambda_param, self.max_span_length)
            end = min(start + span_length, seq_length)
            
            # avoid buffer zones
            span_positions = set(range(start, end))
            if span_positions & buffer_positions:
                continue
            
            new_positions = span_positions - masked_positions
            if new_positions:
                spans.append((start, end))
                masked_positions.update(new_positions)
                
                if start > 0:
                    buffer_positions.add(start - 1)
                if end < seq_length:
                    buffer_positions.add(end)
        
        # use single tokens to reach target rate if necessary
        while len(masked_positions) < target_masked_chars:
            available = [i for i in range(seq_length) 
                        if i not in masked_positions and i not in buffer_positions]
            if not available:
                break
            pos = self._py_rng.choice(available)
            spans.append((pos, pos + 1))
            masked_positions.add(pos)
            
            if pos > 0:
                buffer_positions.add(pos - 1)
            if pos + 1 < seq_length:
                buffer_positions.add(pos + 1)
        
        return sorted(spans)
    
    def _mask_sequence(self, input_ids: List[int]) -> Dict:
        """
        Apply dynamic masking to txt
        
        Parameters:
            input_ids (list[int]) -- original unmasked sequence
            
        Returns:
            dict of masked inputs and labels
        """
        seq_length = len(input_ids)
        
        current_mlm_prob = self.get_current_mlm_prob()
        
        excluded_positions = (
            {i for i, tok in enumerate(input_ids) if tok in self._excluded_token_ids}
            if self._excluded_token_ids else None
        )

        spans = self._select_mask_spans(
            seq_length, current_mlm_prob, excluded_positions
        )
        
        # labels (-100 for unmasked)
        labels = [-100] * seq_length
        
        masked_input_ids = list(input_ids) if not isinstance(input_ids, list) else input_ids.copy()
        
        # BERT-style masking
        for start, end in spans:
            for pos in range(start, end):
                labels[pos] = input_ids[pos]
                rand = self._py_rng.random()
                if rand < self.mask_replace_prob:
                    masked_input_ids[pos] = self.mask_token_id
                elif rand < self.mask_replace_prob + self.random_replace_prob:
                    random_id = self._py_rng.choice(self._random_replacement_ids)
                    masked_input_ids[pos] = random_id
        
        return {
            'input_ids': masked_input_ids,
            'labels': labels,
        }
    
    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate masked batch
        
        Parameters:
            examples (list[dict]) -- unmasked data samples w/ input_ids
        
        Returns:
            batched dict of dynamically masked samples
        """
        self._ensure_rngs()

        if self._py_rng.random() < 0.01:
           current_prob = self.get_current_mlm_prob()
           print(f"Step {self.current_step}: Current MLM prob = {current_prob:.3f}")

        max_length = max(len(ex['input_ids']) for ex in examples)
        
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        
        for example in examples:
            original_input_ids = example['input_ids']
            
            masked_data = self._mask_sequence(original_input_ids)
            
            pad_length = max_length - len(original_input_ids)
            
            input_ids = masked_data['input_ids'] + [self.pad_token_id] * pad_length
            labels = masked_data['labels'] + [-100] * pad_length
            attention_mask = [1] * len(original_input_ids) + [0] * pad_length
            
            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
            batch_attention_mask.append(attention_mask)
        
        result = {
            'input_ids': torch.tensor(batch_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(batch_attention_mask, dtype=torch.long),
            'labels': torch.tensor(batch_labels, dtype=torch.long),
        }
        
        return result


class DatasetForMaskedLM(torch.utils.data.Dataset):
    """
    Wrapper dataset for unmasked input
    """

    def __init__(self, original_dataset):
        self.data = original_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'input_ids': item.get('original_input_ids', item['input_ids']),
        }
