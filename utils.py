import torch
import logging
from flwr.common import NDArrays, Scalar
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
from typing import Optional
import datasets
import evaluate
import numpy as np
from datasets import load_dataset
from hydra import compose, initialize
from flwr_datasets import FederatedDataset
import matplotlib.pyplot as plt
import flwr as fl
from flwr.common import Context
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from typing import Callable, Dict, Tuple
import csv


task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}
logger = logging.getLogger(__name__)

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    
    epsilon: int = field(
        metadata={"help": "Target epsilon"},
    )
    
    accountant: str = field(
        metadata={"help": "Accountant can be moments, prv, ew"},
    )
    
    
    partition_policy: str = field(
        metadata={"help": "Partition policy. It can be: Iid, Linear, Square, or Exp"},
    )

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task, a training/validation file or a dataset name.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )

def get_config(config_name: str):
    with initialize(config_path="./conf", version_base="1.1"):
        cfg = compose(config_name=config_name)

    return cfg
def visualize_partitions(fed_dataset: FederatedDataset):
    _ = fed_dataset.load_partition(0)
    num_partitions = fed_dataset.partitioners['train'].num_partitions
    
    plt.bar(range(num_partitions), [len(fed_dataset.load_partition(i)) for i in range(num_partitions)])
    plt.xticks(range(num_partitions))
    plt.xlabel("Partition ID")
    plt.ylabel("Number of examples")
    plt.title(f"IID partitioning into {num_partitions} partitions")
    plt.savefig("matplotlib.png")

def get_model(model_args,data_args,num_labels):
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )
    return config,tokenizer,model


class HuggingFaceClient(fl.client.NumPyClient):

    def __init__(self,model,train_dataset, eval_dataset, training_args,processing_class,data_collator,task_name):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.training_args = training_args
        self.processing_class = processing_class
        self.data_collator = data_collator
        self.dataset_name = task_name

        #end
    def get_parameters(self,config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        for param, val in zip(state_dict.keys(), parameters):
            state_dict[param].copy_(torch.tensor(val))
        self.model.load_state_dict(state_dict)
        
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        def compute_metrics(p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            preds = np.argmax(preds, axis=1)
            metric = evaluate.load("glue", self.dataset_name)
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            processing_class=self.processing_class,
            compute_metrics = compute_metrics,
            data_collator=self.data_collator
        )
        eval_res = trainer.evaluate()
        print("****")
        print(eval_res)
        return float(eval_res["eval_loss"]), len(self.eval_dataset), {"accuracy": float(eval_res["eval_accuracy"])}

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        def compute_metrics(p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            preds = np.argmax(preds, axis=1)
            metric = evaluate.load("glue", self.dataset_name)
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            processing_class=self.processing_class,
            compute_metrics = compute_metrics,
            data_collator=self.data_collator
        )
        results = trainer.train()
        metrics = {}
        eval_res = trainer.evaluate()
        metrics['eval_loss'] = eval_res['eval_loss']
        metrics['eval_accuracy'] = eval_res['eval_accuracy']
        metrics = {**metrics, "eval_loss": metrics['eval_loss'], "eval_accuracy":  metrics['eval_accuracy']}
        return (
            self.get_parameters({}),
            len(self.train_dataset),
            metrics,
        )

def gen_client_fn(
    fds,
    training_args,
    model_args,
    data_args,
    label_list
) -> Callable[[str], HuggingFaceClient]:  # pylint: disable=too-many-arguments
    """Generate the client function that creates the Flower Clients."""

    def client_fn(context: Context) -> HuggingFaceClient:
        num_labels = len(label_list)
        """Create a Flower client representing a single organization."""
        config,tokenizer,model = get_model(model_args,data_args,num_labels)
        # Let's get the partition corresponding to the i-th client
        partition_id = int(context.node_config["partition-id"])
        partition = fds.load_partition(partition_id, "train")
        partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
            # Padding strategy
        if data_args.pad_to_max_length:
            padding = "max_length"
        else:
            # We will pad later, dynamically at batch creation, to the max sequence length in each batch
            padding = False
            # Some models have set the order of the labels to use, so let's make sure we do use it.
        label_to_id = None
        if (
            model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
            and data_args.task_name is not None
        ):
                # Some have all caps in their config, some don't.
            label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
            if sorted(label_name_to_id.keys()) == sorted(label_list):
                label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
            else:
                logger.warning(
                    "Your model seems to have been trained with labels, but they don't match the dataset: "
                    f"model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(label_list)}."
                    "\nIgnoring the model labels as a result.",
                )
        elif data_args.task_name is None:
                label_to_id = {v: i for i, v in enumerate(label_list)}
        if label_to_id is not None:
            model.config.label2id = label_to_id
            model.config.id2label = {id: label for label, id in config.label2id.items()}
        elif data_args.task_name is not None:
            model.config.label2id = {l: i for i, l in enumerate(label_list)}
            model.config.id2label = {id: label for label, id in config.label2id.items()}
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the "
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
        def preprocess_function(examples):
            # Tokenize the texts
            args = (
                (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
            )
            result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)
            # Map labels to IDs (not necessary for GLUE tasks)
            if label_to_id is not None and "label" in examples:
                result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
            return result
        with training_args.main_process_first(desc="dataset map pre-processing"):
            partition_train_test = partition_train_test.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
        #Data_collator
        if data_args.pad_to_max_length:
            data_collator = default_data_collator
        elif training_args.fp16:
            data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
        else:
            data_collator = None
        return HuggingFaceClient(
            model,
            partition_train_test["train"],
            partition_train_test["test"],
            training_args,
            tokenizer,
            data_collator,
            data_args.task_name).to_client()
    return client_fn

#Server
# def get_on_fit_config():
#     def fit_config_fn(server_round: int):
#         fit_config = {"current_round": server_round}
#         return fit_config

#     return fit_config_fn


def fit_weighted_average(metrics):
    """Aggregation function for (federated) evaluation metrics."""
    # Multiply accuracy of each client by number of examples used
    losses = [num_examples * m["eval_loss"] for num_examples, m in metrics]
    # acuuracy = [num_examples * m["eval_accuracy"] for num_examples, m in metrics]

    examples = [num_examples for num_examples, _ in metrics]
    # print(metrics)
    # Aggregate and return custom metric (weighted average)
    return {"eval_loss": sum(losses) / sum(examples)}

def get_evaluate_fn(model_args, save_every_round, total_round, save_path,training_args,eval_dataset,tokenizer,data_collator,metric,model_performance_file
):
    """Return an evaluation function for saving global model."""


    def set_parameters(model, parameters):
        state_dict = model.state_dict()
        for param, val in zip(state_dict.keys(), parameters):
            state_dict[param].copy_(torch.tensor(val))
            model.load_state_dict(state_dict)

    def evaluate(server_round: int, parameters, config) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        loss = 0
        accuracy = 0
        # Save model
        if server_round != 0 and (
            server_round == total_round or server_round % save_every_round == 0
        ):
            # Init model
            model = AutoModelForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                token=model_args.token,
                trust_remote_code=model_args.trust_remote_code,
                ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
                )
            set_parameters(model, parameters)
            def compute_metrics(p: EvalPrediction):
                preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
                preds = np.argmax(preds, axis=1)
                result = metric.compute(predictions=preds, references=p.label_ids)
                if len(result) > 1:
                    result["combined_score"] = np.mean(list(result.values())).item()
                return result
            trainer = Trainer(
                    model=model,
                    args=training_args,
                    eval_dataset=eval_dataset,
                    compute_metrics=compute_metrics,
                    processing_class=tokenizer,
                    data_collator=data_collator,
            )
            eval_res = trainer.evaluate()
            loss = eval_res["eval_loss"]
            accuracy = eval_res["eval_accuracy"]
            model.save_pretrained(f"{save_path}/Model_{server_round}")
            row = [server_round,accuracy,eval_res]
            with open(model_performance_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(row)  # Writing a new row in each iteration
            # print("FINAL IS HERE")
            # print(eval_res)
        # return float(loss), {"accuracy":accuracy}
        return 0.0, {}
    return evaluate
