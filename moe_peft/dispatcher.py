import logging
import random
import sys
from abc import abstractmethod
from typing import Callable, Dict, List

import datasets

from .common import InputData, LLMBatchConfig, LLMModelInput, Masks, Tokens
from .tokenizer import Tokenizer


class Event:
    __callback_list: List[Callable] = None

    def __init__(self):
        self.__callback_list = []

    def register(self, func: Callable) -> "Event":
        self.__callback_list = [func] + self.__callback_list
        return self

    def activate(self, **kwargs) -> bool:
        for func in self.__callback_list:
            if func(**kwargs):
                return True

        return False


def load_dataset(data_path: str):
    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        return datasets.load_dataset("json", data_files=data_path)
    else:
        if ":" in data_path:
            result = data_path.split(":")
            return datasets.load_dataset(result[0], result[1])
        else:
            return datasets.load_dataset(data_path)


class TrainTask:
    tokenizer_: Tokenizer = None

    adapter_name_: str = ""
    data_path_: str = ""
    dataload_function_: Callable = None
    train_token_data_: List[InputData] = None

    # train parameter
    total_epoch_num_: int = -1
    max_train_batch_size_: int = -1
    max_train_micro_batch_size_: int = -1
    max_test_batch_size_: int = -1

    train_cutoff_len_: int = -1
    group_by_length_: bool = False

    # count the stat of train and test data
    epoch_cnt_: int = 1
    next_train_data_start_idx_: int = 0
    next_test_data_start_idx_: int = 0

    def __init__(
        self,
        tokenzer: Tokenizer,
        adapter_name: str,
        dataload_function: Callable,
        total_epoch_num: int,
        max_train_batch_size: int,
        max_train_micro_batch_size: int,
        train_cutoff_len: int = 256,
        group_by_length: bool = True,
    ):
        self.tokenizer_ = tokenzer
        self.adapter_name_ = adapter_name
        self.dataload_function_ = dataload_function
        self.total_epoch_num_ = total_epoch_num
        self.max_train_batch_size_ = max_train_batch_size
        self.max_train_micro_batch_size_ = max_train_micro_batch_size
        self.train_cutoff_len_ = train_cutoff_len
        self.group_by_length_ = group_by_length

    def load_data(self):
        self.train_token_data_ = self.dataload_function_(self.tokenizer_)
        max_train_tokens_len = 0
        for data in self.train_token_data_:
            max_train_tokens_len = max(max_train_tokens_len, len(data.tokens))
            if len(data.tokens) > self.train_cutoff_len_:
                data.tokens = data.tokens[: self.train_cutoff_len_]

        logging.info(
            f"Max train tokens length: {max_train_tokens_len}/{self.train_cutoff_len_}"
        )
        if self.group_by_length_:
            self.train_token_data_.sort(key=lambda x: len(x.tokens), reverse=True)
        else:
            random.shuffle(self.train_token_data_)

    def is_train_done(self):
        if self.epoch_cnt_ <= self.total_epoch_num_:
            return False
        return True

    def is_test_done(self):
        if self.next_test_data_start_idx_ < len(self.test_token_data_):
            return False
        return True

    def reset_test_status(self):
        self.next_test_data_start_idx_ = 0

    # reentry function
    def get_train_deta_max_seq_len(self) -> int:
        start_idx = self.next_train_data_start_idx_
        assert start_idx < len(self.train_token_data_)
        # in this strategy must sort
        return len(self.train_token_data_[start_idx].tokens)

    # non reentry function
    def get_train_data(self) -> List[InputData]:
        start_idx = self.next_train_data_start_idx_
        end_idx = start_idx + self.max_train_micro_batch_size_

        ret_data = self.train_token_data_[start_idx:end_idx]

        logging.info(f"{self.adapter_name_} train data:")
        logging.info(
            f"    epoch: {self.epoch_cnt_}/{self.total_epoch_num_} \
            step in epoch: {start_idx}/{len(self.train_token_data_)}"
        )

        self.next_train_data_start_idx_ += self.max_train_micro_batch_size_
        if self.next_train_data_start_idx_ >= len(self.train_token_data_):
            self.next_train_data_start_idx_ = 0
            self.epoch_cnt_ += 1

        return ret_data


class DispatcherConfig:
    @abstractmethod
    def dispatcher_context(self) -> Dict[str, any]:
        return {}


class Dispatcher:
    config_ = None
    tokenizer_: Tokenizer = None

    # all train task
    ready_train_task_: List[TrainTask] = None
    running_train_task_: List[TrainTask] = None
    done_train_task_: List[TrainTask] = None

    # train task in event
    train_task_in_event_: Event = None
    train_task_out_event_: Event = None

    # the number of max candidate training lora model
    # can chose train data from this dataset
    train_lora_candidate_num_: int = 0
    # the number of simultaneously train lora model
    train_lora_simultaneously_num_: int = 0

    strategy_: str = ""

    def __init__(
        self,
        tokenizer: Tokenizer,
        configs: List[DispatcherConfig],
        max_concurrent_jobs: int = None,
        strategy: str = "optim",
        cutoff_len: int = 256,
    ) -> None:
        if max_concurrent_jobs is None:
            max_concurrent_jobs = len(configs)

        self.tokenizer_ = tokenizer

        self.ready_train_task_ = []
        self.running_train_task_ = []
        self.done_train_task_ = []

        self.train_task_in_event_ = Event()
        self.train_task_out_event_ = Event()

        self.train_lora_candidate_num_ = sys.maxsize
        self.train_lora_simultaneously_num_ = max_concurrent_jobs
        self.strategy_ = strategy

        # create ready task
        for config_class in configs:
            kwargs = config_class.dispatcher_context()
            self.ready_train_task_.append(
                TrainTask(
                    tokenzer=self.tokenizer_, train_cutoff_len=cutoff_len, **kwargs
                )
            )

    def optim_dispatch_strategy(self) -> Dict[str, List[InputData]]:
        task_len = {}
        for idx, task in enumerate(self.running_train_task_):
            task_len[idx] = task.get_train_deta_max_seq_len()
        # sort to get the seq most similar data
        task_len = sorted(task_len.items(), key=lambda x: x[1], reverse=True)
        # find the mini diff
        min_need_pad_len = sys.maxsize
        win_start_idx = 0
        for sidx in range(0, len(task_len) - self.train_lora_simultaneously_num_ + 1):
            win = task_len[sidx : sidx + self.train_lora_simultaneously_num_]
            need_pad_len = 0
            for i in range(1, len(win)):
                # aligin to the max seq len
                need_pad_len += abs(win[i][1] - win[0][1])
            if need_pad_len < min_need_pad_len:
                min_need_pad_len = need_pad_len
                win_start_idx = sidx
        # the result is win_start_idx
        result_win = task_len[
            win_start_idx : win_start_idx + self.train_lora_simultaneously_num_
        ]
        ret_train_data = {}
        for result_task_len in result_win:
            task_idx = result_task_len[0]
            ret_train_data[self.running_train_task_[task_idx].adapter_name_] = (
                self.running_train_task_[task_idx].get_train_data()
            )

        return ret_train_data

    def none_dispatch_strategy(self) -> Dict[str, List[InputData]]:
        ret_train_data = {}
        cnt = 0
        for task in self.running_train_task_:
            assert not task.is_train_done()
            if cnt >= self.train_lora_simultaneously_num_:
                break
            ret_train_data[task.adapter_name_] = task.get_train_data()
            cnt += 1
        return ret_train_data

    def check_task_done(self) -> bool:
        if len(self.ready_train_task_) == 0 and len(self.running_train_task_) == 0:
            return True
        return False

    def check_test_done(self) -> bool:
        for task in self.running_train_task_:
            if task.is_train_done():
                return False
        return True

    def reset_test_task(self):
        for task in self.running_train_task_:
            task.reset_test_status()

    # ready task -> running task
    def __dispatch_task_in(self):
        assert len(self.running_train_task_) <= self.train_lora_candidate_num_
        if len(self.running_train_task_) == self.train_lora_candidate_num_:
            return
        # chose task into running
        while (
            len(self.running_train_task_) < self.train_lora_candidate_num_
            and len(self.ready_train_task_) > 0
        ):
            # TODO to dispatch task
            task = self.ready_train_task_.pop(0)
            # to lazy load data
            task.load_data()
            self.train_task_in_event_.activate(task=task)
            self.running_train_task_.append(task)

    # running task -> done task
    def __dispatch_task_out(self):
        for task in self.running_train_task_:
            if task.is_train_done():
                self.train_task_out_event_.activate(task=task)
                self.done_train_task_.append(task)

        self.running_train_task_ = [
            task for task in self.running_train_task_ if not task.is_train_done()
        ]

    def get_test_data(self) -> LLMModelInput:
        pass

    def get_train_data(self) -> LLMModelInput:
        self.__dispatch_task_in()

        # get task train data
        all_train_data: Dict[str, List[InputData]] = {}
        if self.strategy_ == "none":
            all_train_data = self.none_dispatch_strategy()
        elif self.strategy_ == "optim":
            all_train_data = self.optim_dispatch_strategy()
        else:
            raise "unkown strategy"

        batch_seq_len: int = -1
        # to align batch token data
        for adapter in all_train_data:
            for data in all_train_data[adapter]:
                batch_seq_len = max(batch_seq_len, len(data.tokens))

        # all prompts and tokens / config
        batch_tokens: List[Tokens] = []
        attention_masks: List[Masks] = []
        batch_labels: List[List] = []
        lora_batch_data_config: List[LLMBatchConfig] = []

        # batch the all adapter data
        adapter_start_idx: int = 0
        for adapter in all_train_data:
            adapter_end_idx: int = adapter_start_idx + len(all_train_data[adapter])
            for data in all_train_data[adapter]:
                tokens: Tokens = data.tokens.copy()
                pad_side = self.tokenizer_.padding_side_
                assert pad_side == "right" or pad_side == "left"
                # pad the tokens to align
                while len(tokens) < batch_seq_len:
                    if pad_side == "right":
                        tokens.append(self.tokenizer_.pad_id_)
                    else:
                        tokens.insert(0, self.tokenizer_.pad_id_)
                batch_tokens.append(tokens)
                attention_masks.append(self.tokenizer_.mask_from(tokens))
                labels = data.labels
                if labels is None:
                    labels = tokens.copy()
                else:
                    labels = labels.copy()
                batch_labels.append(labels)

            lora_batch_data_config.append(
                LLMBatchConfig(
                    adapter_name_=adapter,
                    batch_start_idx_=adapter_start_idx,
                    batch_end_idx_=adapter_end_idx,
                )
            )
            adapter_start_idx = adapter_end_idx

        self.__dispatch_task_out()

        return LLMModelInput(
            batch_configs_=lora_batch_data_config,
            batch_tokens_=batch_tokens,
            batch_labels_=batch_labels,
            batch_masks_=attention_masks,
            gradient_checkpoint_="recompute",
        )
