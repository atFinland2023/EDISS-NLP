import os
import pandas as pd
from multiprocessing import Pool, cpu_count
import configparser
from tqdm import tqdm
from src.utils.utils import *
from src.utils.tokenize_with_huggingface import *
from src.utils.customized_configparser import *
from src.utils.logger import *
from src.model.LSTM_GPT import *
from src.model.LSTM import *
from src.model.general_models import *
from functools import partial

def all_in_one(logger, parameters):
    if not os.path.exists(parameters['data_path']+'/'+ parameters['preprocessed_data']):
        df_raw = pd.read_csv(parameters['data_path']+'/'+ parameters['raw_data'], sep='\t')
        
        # Apply preprocess_text() to each row using multiprocessing
        num_processes = cpu_count()  # Number of CPU cores
        logger.info(f'{num_processes} cores are using to process the data for accelerating processing time.')

        # Use partial to fix the function signature
        partial_preprocess_wrapper = partial(preprocess_wrapper)

        with Pool(processes=num_processes) as pool:
            result = list(tqdm(pool.imap(partial_preprocess_wrapper, df_raw.iterrows()), total=len(df_raw)))

        # Convert the list of processed rows back to a DataFrame
        df = pd.DataFrame(result)
        df = df[['sentiment_label', 'tweet_text_processed']]
        df = df.dropna()
        df.to_csv(parameters['data_path']+'/'+ parameters['preprocessed_data'], index=False)
    else: 
        df =  pd.read_csv(parameters['data_path']+'/'+ parameters['preprocessed_data'])
    
    # Log the original and processed tweet content for the first 10 samples
    for index, row in df.head(5).iterrows():
        logger.info(f"For {index} sample, the processed tweet content is: {row['tweet_text_processed']}")
    
    # drop NA
    df.dropna(subset=['tweet_text_processed'], inplace=True)
    
    # Get word cloud
    get_word_cloud(df, logger, parameters)
        
    # Train General Models
    test_and_evaluate_models(df, logger, parameters)
    
    #Tokenizing
    tokenized_data, vocab_size = tokenize_with_huggingface(df, logger, parameters)
    #train GPT
    if 'gpt' in parameters['tokenizer']: 
        train_validate_test_LSTM_GPT(tokenized_data, vocab_size, logger, parameters)
    # Train BERT
    else: 
        train_validate_test_LSTM(tokenized_data, vocab_size, logger, parameters)
    
if __name__ == "__main__":

    # Read configuration from file
    config = configparser.ConfigParser()
    config.read('config.ini')
    
    # Access preprocessing variables and remove comments
    parameters = remove_comments_and_convert(config, 'global')

    # Create a logger
    logger = get_logger('results.log', parameters['log_path'])
    logger.info(f'==========New Line==========')
    print(f"Check log later in {parameters['log_path']}")
    # Call the all_in_one function with parsed arguments
    all_in_one(logger, parameters)
