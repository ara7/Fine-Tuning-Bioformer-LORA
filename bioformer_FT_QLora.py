#Author: Ara, Lena
#Description: HPC-Optimized Supervised Fine-Tuning (SFT) of Bioformer using QLoRA and Multiprocessed Data Pipelines.

import pandas as pd
import numpy as np
import random
import torch
import time
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, roc_curve, auc, precision_recall_fscore_support
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, AutoTokenizer, AutoModel, BitsAndBytesConfig
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch import nn
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# Data Loading & Preprocessing

df_train = pd.read_csv('../train_delirium_new_work_done.csv')
df_test = pd.read_csv('../test_delirium_new_work_done.csv')

df_train = df_train.dropna(subset=["processed_extracted"])
df_test = df_test.dropna(subset=["processed_extracted"])

print('after dropping null')
print('Train', df_train.groupby(["any_inpt_delirium"]).size())
print('Test', df_test.groupby(["any_inpt_delirium"]).size())

df_train_case = df_train[df_train['any_inpt_delirium'] == 1]
df_train_control= df_train[df_train['any_inpt_delirium'] == 0]
print(len(df_train_case), len(df_train_case.SID.unique()))
print(len(df_train_control), len(df_train_control.SID.unique()))

#df_train= pd.concat([df_train_case[:20], df_train_control[:20]], axis=0)


model_path = "/geode3/projects/path/paper_3_again/saint/models_huggingface/bioformer-16L"
tokenizer = AutoTokenizer.from_pretrained(model_path)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPUs available.')
    print('Device name:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

X_train = df_train[['processed_extracted','SID']].to_numpy()
y_train = df_train.any_inpt_delirium.to_numpy()

MAX_LEN = 1024

def preprocessing_for_bert(data):
    input_ids = []
    attention_masks = []
    numerical_sid = []
    for sent in data:
        encoded_sent = tokenizer.encode_plus(
            text=sent[0],
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding="max_length",
            return_attention_mask=True,
            truncation=True
        )
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))
        numerical_sid.append(sent[1])

    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    numerical_sid = torch.tensor(numerical_sid)
    return input_ids, attention_masks, numerical_sid

print('Tokenizing data..')
train_inputs, train_masks, numerical_feat = preprocessing_for_bert(X_train)
train_labels = torch.tensor(y_train)
print('Done tokenizing')


# HPC OPTIMIZED: Multiprocessing DataLoader

batch_size = 32
train_data = TensorDataset(train_inputs, train_masks, numerical_feat, train_labels)
train_sampler = RandomSampler(train_data)


train_dataloader = DataLoader(
    train_data,
    sampler=train_sampler,
    batch_size=batch_size,
    num_workers=8,
    pin_memory=True     #for V100/SLURM optimization
)


# QLoRA Model Definition

class BertClassifier(nn.Module):
    def __init__(self):
        super(BertClassifier,self).__init__()
        D_in, H, D_out = 384, 100, 2
        self.model_path = "/geode3/projects/path/paper_3_again/saint/models_huggingface/bioformer-16LL"

        #check if gpu available
        self.use_gpu = torch.cuda.is_available()
        if self.use_gpu:
            print('GPU detected! Initializing QLora for V100')

            # Hardware-Aware Quantization (V100 requires float16, NOT bfloat16)
            compute_dtype = torch.float16
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )

            # Load Base Model in 4-bit
            self.bert = AutoModel.from_pretrained(
                self.model_path,
                quantization_config=bnb_config,
                device_map={"": torch.cuda.current_device()} if torch.cuda.is_available() else "auto"
            )

            # Prepare for QLoRA Training
            self.bert = prepare_model_for_kbit_training(self.bert)

            # Apply LoRA Adapters to attention modules
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["query", "value"],
                lora_dropout=0.05,
                bias="none",
                task_type="FEATURE_EXTRACTION"
            )
            self.bert = get_peft_model(self.bert, lora_config)
            self.bert.print_trainable_parameters() # Prints % of parameters being trained
        else:
            print('Initializing in CPU/ Debug Mode. No GPU found')
            self.bert = AutoModel.from_pretrained(self.model_path)
            # No LoRA/Quantization applied here so it works on CPU

        # Custom Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Linear(H, D_out)
        )

    def forward(self, input_ids, attention_mask, numerical_feat):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        a = outputs.last_hidden_state
        last_hidden_state_cls = a[:,0,:]

        logits = self.classifier(last_hidden_state_cls)
        return logits, numerical_feat

def initialize_model(epochs=2):
    bert_classifier = BertClassifier()
    # Move classification head to GPU (Base model is already mapped)
    bert_classifier.classifier.to(device)

    # Only pass parameters that require gradients to the optimizer
    optimizer = AdamW(filter(lambda p: p.requires_grad, bert_classifier.parameters()), lr=5e-5, eps=1e-8)

    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    return bert_classifier, optimizer, scheduler

loss_fn = nn.CrossEntropyLoss()

def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def save_model(network, epoch_label):
    save_filename = 'model_%s.pth' % epoch_label
    save_path = os.path.join('./bioformer_new_work_QLora', save_filename)
    os.makedirs('./bioformer_new_work_QLora', exist_ok=True)
    torch.save(network.state_dict(), save_path) # Saving state_dict is safer for PEFT models

def train(model, train_dataloader, epochs=7):
    print("Start training...\n")
    for epoch_i in range(epochs):
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Tr Acc':^9} | {'Elapse':^9}")
        print("-"*60)
        t0_epoch, t0_batch = time.time(), time.time()
        total_loss, batch_loss, batch_counts = 0,0,0
        model.train()

        for step, batch in enumerate(train_dataloader):
            batch_counts += 1
            b_input_ids, b_attn_mask, b_num, b_labels = tuple(t.to(device) for t in batch)
            b_labels = b_labels.to(torch.int64)

            model.zero_grad()
            logits, sid = model(b_input_ids, b_attn_mask, b_num)

            loss = loss_fn(logits, b_labels)
            batch_loss += loss.item()
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            # Print loss values and time elapsed
            if (step % 20 == 0 and step != 0) or (step == len(train_dataloader)-1):
                # Calculate accuracy on the current batch only, NOT the whole dataset
                preds = torch.argmax(logits, dim=1).flatten()
                tr_accuracy = (preds == b_labels).cpu().numpy().mean() * 100

                time_elapsed = time.time() - t0_batch
                print(f"{epoch_i+1:^7} | {step:^7} | {batch_loss / batch_counts :^12.6f} | {tr_accuracy:^9.2f} | {time_elapsed:^9.2f}")

                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        avg_train_loss = total_loss / len(train_dataloader)
        print("-"*60)
        print(f"{epoch_i+1:^7} | {'-':^7} | {avg_train_loss :^12.6f} | {'-':^9} | {time.time() - t0_epoch:^9.2f}")
        print("-"*60)
        save_model(model, epoch_i)

    print("Training complete!")

set_seed(42)
bert_classifier, optimizer, scheduler = initialize_model(epochs=10)
train(bert_classifier, train_dataloader, epochs=10)
