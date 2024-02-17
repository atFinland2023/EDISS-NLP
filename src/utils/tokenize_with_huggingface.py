import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# PyTorch dataset for text vectorization
class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item["tweet_text"],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        sentiment_label = item['sentiment_label']
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'sentiment_label': sentiment_label}


# Tokenization using Hugging Face tokenizer
def tokenize_with_huggingface(data, logger):
    
    max_length = data["tweet_text_processed"].apply(lambda x: len(x.split())).max()
    
    logger.info(f"Number of Tokenized features are {max_length}")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # Get the vocabulary size
    vocab_size = tokenizer.vocab_size
    tokenized_data = tokenizer(
        [row["tweet_text_processed"] for _, row in data.iterrows()],
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    # Encode sentiment labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(data["sentiment_label"])

    # Add encoded labels to the tokenized data
    tokenized_data["sentiment_label"] = encoded_labels

    return tokenized_data, vocab_size


# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# token_ids = [101,  1030, 10777,  9001,  4931,  6203,  1010,  3407,  5958,  2000,
#           2017,  2525,  2018,  2115,  5785,  1005,  1055,  4605,  2005,  6265,
#            102]
# tokens = tokenizer.convert_ids_to_tokens(token_ids)

# print(tokens)
# # PyTorch dataset for text vectorization
# text_dataset = TextDataset(tokenized_data, tokenizer, max_length=max_length)

# # Dataloader for batching
# dataloader = DataLoader(text_dataset, batch_size=2, shuffle=False)

# # Iterate through batches
# for batch in dataloader:
#     input_ids = batch['input_ids']
#     attention_mask = batch['attention_mask']
#     sentiment_labels = batch['sentiment_label']

#     # Print batched tensors and sentiment labels
#     print("Batched input_ids:", input_ids)
#     print("Batched attention_mask:", attention_mask)
#     print("Sentiment Labels:", sentiment_labels)
