""" Train Bert on IMDB Sentiment dataset"""

import os
import time
import datetime
import random
import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup

# directories
DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")

# hyperparameters
BATCH_SIZE = 64
# N_EPOCHS = 4
N_EPOCHS = 2

# set seed
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# global variables
device = torch.device("cuda")


def _load_train(label_encoder, tokenizer):
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df = df.head(10)    # temporary

    # encode label
    df.sentiment = label_encoder.fit_transform(df.sentiment)

    sentences = df.review.values
    labels = df.sentiment.values

    input_ids = []
    attention_masks = []
    # for sent in sentences:
    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
            sent,
            add_special_tokens=True,
            max_length=128,
            pad_to_max_length=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    train_dataset = TensorDataset(input_ids, attention_masks, labels)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=BATCH_SIZE
    )

    return train_dataloader


def _load_val(label_encoder, tokenizer):
    df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    input_ids = []
    attention_masks = []

    # process sentences
    for sent in df.review.values:
        encoded_dict = tokenizer.encode_plus(
            sent,
            add_special_tokens=True,
            max_length=128,
            pad_to_max_length=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    # process labels
    labels = label_encoder.transform(df.sentiment)
    labels = torch.tensor(labels)

    val_dataset = TensorDataset(input_ids, attention_masks, labels)
    validation_dataloader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=BATCH_SIZE
    )

    return validation_dataloader


def _load_data(tokenizer):
    """ """

    # label encoder
    label_encoder = LabelEncoder()

    train_dataloader = _load_train(label_encoder, tokenizer)
    val_dataloader = _load_val(label_encoder, tokenizer)

    return train_dataloader, val_dataloader


# start train
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    print(pred_flat)
    print(labels_flat)

    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def _eval(model, validation_dataloader):
    """ """
    t0 = time.time()

    model.eval()

    total_eval_accuracy = 0
    total_eval_loss = 0

    for batch in validation_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        with torch.no_grad():
            loss, logits = model(b_input_ids,
                                 token_type_ids=None,
                                 attention_mask=b_input_mask,
                                 labels=b_labels)

    total_eval_loss += loss.item()

    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()

    total_eval_accuracy += flat_accuracy(logits, label_ids)

    # Report the final accuracy for this validation run.
    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(validation_dataloader)

    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)

    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    return avg_val_loss, avg_val_accuracy, validation_time


def train():
    # load Bert tokenizer
    print('Loading BertTokenizer ...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    # load data
    train_dataloader, validation_dataloader = _load_data(tokenizer)

    # set model
    print('Loading BertForSequenceClassification ...')
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False
    )
    model.cuda()

    # set optimizer
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    total_steps = len(train_dataloader) * N_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # start train
    training_stats = []
    total_t0 = time.time()

    for epoch_i in range(N_EPOCHS):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, N_EPOCHS))
        print('Training...')

        t0 = time.time()

        total_train_loss = 0

        model.train()

        for step, batch in enumerate(train_dataloader):
            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()
            loss, logits = model(b_input_ids,
                                 token_type_ids=None,
                                 attention_mask=b_input_mask,
                                 labels=b_labels)

            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))

        print("")
        print("Running Validation...")
        avg_val_loss, avg_val_accuracy, validation_time = _eval(model, validation_dataloader)


        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))


    # display training statistics
    pd.set_option('precision', 2)
    df_stats = pd.DataFrame(data=training_stats)
    df_stats = df_stats.set_index('epoch')
    print(df_stats)

    output_dir = "/home/dkim/Projects/KD/train_bert/model_saved"
    print("Saving model to %s" % output_dir)
    model_to_save = model.module if hasattr(model, 'mdule') else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    train()