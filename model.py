import pandas as pd
import numpy as np
from transformers import pipeline

# load the CSV files into dataframes
messages_df = pd.read_csv("chat_messages_1.csv")
accounts_df = pd.read_csv("accounts.csv")

# merge the two dataframes on account_id
df = pd.merge(messages_df, accounts_df, on='account_id', how='left')

# define the topics to classify the messages into
topics = ["GENERAL_RISK", "BULLYING", "VIOLENCE", "RELATIONSHIP_SEXUAL_CONTENT", "VULGARITY",
          "DRUGS_ALCOHOL", "IN_APP", "ALARM", "FRAUD", "HATE_SPEECH"]

# initialize the DistilBERT model for sequence classification
model = pipeline('text-classification',
                 model='distilbert-base-uncased', return_all_scores=True)

# create a new dataframe to store the results
result_df = pd.DataFrame(columns=['account_id', 'raw_message'] + topics)

# iterate through each row of the dataframe and classify the messages into each topic
for index, row in df.iterrows():
    message = row["raw_message"]
    scores = model(message, labels=topics)
    scores_dict = {topic: score for topic,
                   score in zip(topics, scores[0]["scores"])}
    row_dict = {"account_id": row["account_id"], "raw_message": message}
    row_dict.update(scores_dict)
    result_df = result_df.append(row_dict, ignore_index=True)

# save the results to a new CSV file
result_df.to_csv("classified_messages.csv", index=False)
