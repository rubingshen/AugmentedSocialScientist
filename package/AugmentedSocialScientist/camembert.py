import os
import time
import datetime
import random
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from scipy.special import softmax
from nltk.tokenize import sent_tokenize

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,precision_recall_fscore_support

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import CamembertTokenizer, CamembertConfig,\
                         AdamW, get_linear_schedule_with_warmup,\
                         WEIGHTS_NAME, CONFIG_NAME
from transformers import CamembertForSequenceClassification

#device = torch.device("cuda")

# If there's a GPU available...
if torch.cuda.is_available():

    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use GPU {}:'.format(torch.cuda.current_device()), torch.cuda.get_device_name(torch.cuda.current_device()))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


tokenizer = CamembertTokenizer.from_pretrained('camembert-base')

def encode(sentences, labels=None, tokenizer=tokenizer, batch_size=32, progress_bar=True):
    input_ids = []
    if progress_bar:
        sent_loader = tqdm(sentences)
    else:
        sent_loader = sentences
    for sent in sent_loader:
        # `encode` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.  
        encoded_sent = tokenizer.encode(sent,                       # Sentence to encode.
                                        add_special_tokens = True   # Add '[CLS]' and '[SEP]'
                                        #max_length = 128,          # Truncate all sentences.
                                        #return_tensors = 'pt',     # Return pytorch tensors.
                                       )
        input_ids.append(encoded_sent)
        
    MAX_LEN = min(max([len(sen) for sen in input_ids]), 512)

    # Pad our input tokens with value 0.
    # "post" indicates that we want to pad and truncate at the end of the sequence, as opposed to the beginning.
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", 
                              value=0, truncating="post", padding="post")

    # Create attention masks
    attention_masks = []
    if progress_bar:
        input_loader = tqdm(input_ids)
    else:
        input_loader = input_ids
    for sent in input_loader:
        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id > 0) for token_id in sent]

        attention_masks.append(att_mask)
    
    if labels is None:
        # Convert to pytorch tensors
        inputs_tensors = torch.tensor(input_ids)
        masks_tensors = torch.tensor(attention_masks)

        # Create the DataLoader 
        data = TensorDataset(inputs_tensors, masks_tensors)
        sampler = SequentialSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
        return dataloader        
    else:
        # Convert to pytorch tensors
        inputs_tensors = torch.tensor(input_ids)
        masks_tensors = torch.tensor(attention_masks)
        labels_tensors = torch.tensor(labels)

        # Create the DataLoader 
        data = TensorDataset(inputs_tensors, masks_tensors, labels_tensors)
        sampler = SequentialSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
        return dataloader
    
def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded)) # Format as hh:mm:ss

def run_training(train_dataloader, 
                 test_dataloader, 
                 n_epochs=3, 
                 lr=5e-5, 
                 random_state=2018, 
                 save_model_as=None):
    
    # Unpack all test labels for evaluation
    test_labels = []
    for batch in test_dataloader:
        test_labels += batch[2].numpy().tolist()
    num_labels = np.unique(test_labels).size
    
    # Set the seed value all over the place to make this reproducible.
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)
    
    # Load model
    model = CamembertForSequenceClassification.from_pretrained("camembert-base", 
                                                               num_labels = num_labels, 
                                                               output_attentions = False, 
                                                               output_hidden_states = False)
    # Tell pytorch to run this model on the GPU.
    if torch.cuda.is_available():
        model.cuda()
    
    optimizer = AdamW(model.parameters(), lr = lr, eps = 1e-8)
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=len(train_dataloader) * n_epochs)
    #total number of training steps is number of batches * number of epochs.

    # Store the average loss after each epoch so we can plot them.
    train_loss_values = []
    test_loss_values = []
    
    for i_epoch in range(n_epochs):
    # each epoch
        # ========================================
        #               Training
        # ========================================
        # Perform one full pass over the training set.
        
        print("")
        print('======== Epoch {:} / {:} ========'.format(i_epoch + 1, n_epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total train loss for this epoch.
        total_train_loss = 0
        total_test_loss = 0  # Reset the total train loss for this epoch.


        # Put the model in training mode
        model.train()

        for step, train_batch in enumerate(train_dataloader):
        # each batch
            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            b_inputs = train_batch[0].to(device)
            b_masks = train_batch[1].to(device)
            b_labels = train_batch[2].to(device)

            # Clear any previously calculated gradients before performing a
            model.zero_grad()        

            # Perform a forward pass (evaluate the model on this training batch).
            # This will return the loss (rather than the model output) because we have provided the `labels`.
            outputs = model(b_inputs, token_type_ids=None, attention_mask=b_masks,labels=b_labels)

            # The call to `model` always returns a tuple, so we need to pull the loss value out of the tuple.
            loss = outputs[0]

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value 
            # from the tensor.
            total_train_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1. This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()
            
            
        # Calculate the average loss over the training data batches.
        avg_train_loss = total_train_loss / len(train_dataloader)            

        # Store the loss value for plotting the learning curve.
        train_loss_values.append(avg_train_loss)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training took: {:}".format(format_time(time.time() - t0)))

        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on our validation set.

        print("")
        print("Running Validation...")

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently during evaluation.
        model.eval()

        logits_complete = [] # store logits of each batch

        # Evaluate data for one epoch
        for test_batch in test_dataloader:
            # Add batch to GPU
            b_inputs = test_batch[0].to(device)
            b_masks = test_batch[1].to(device)
            b_labels = test_batch[2].to(device)

            # Telling the model not to compute or store gradients, saving memory and speeding up validation
            with torch.no_grad():        
                # Forward pass, calculate test loss and logit predictions.
                # token_type_ids = None : it's not 2-sentences task
                outputs = model(b_inputs, token_type_ids=None, attention_mask=b_masks,labels=b_labels)

            loss = outputs[0].item() # get loss
            logits = outputs[1]      # get logits
            
            # Move logits CPU
            logits = logits.detach().cpu().numpy()
            #labels_id = b_labels.to('cpu').numpy()
            
            total_test_loss += loss
            logits_complete.append(logits)
            
        logits_complete = np.concatenate(logits_complete)
        
        # Calculate the average loss over the test data batches.
        avg_test_loss = total_test_loss / len(test_dataloader)            
        # Store the loss value for plotting the learning curve.
        test_loss_values.append(avg_test_loss)
        
        print("")
        print("  Average test loss: {0:.2f}".format(avg_test_loss))
        print("  Validation took: {:}".format(format_time(time.time() - t0)))
        print(classification_report(test_labels, np.argmax(logits_complete, axis=1).flatten()))
        score = precision_recall_fscore_support(test_labels, np.argmax(logits_complete, axis=1).flatten())


    
        
    # End of all epochs
    print("")
    print("Training complete!")
    
    
    if save_model_as is not None:
        # SAVE
        output_dir = "./models/{}".format(save_model_as)
        try:
            os.makedirs(output_dir)
        except:
            pass
        # Step 1: Save a model, configuration and vocabulary that you have fine-tuned

        # If we have a distributed model, save only the encapsulated model
        # (it was wrapped in PyTorch DistributedDataParallel or DataParallel)
        model_to_save = model.module if hasattr(model, 'module') else model

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        tokenizer.save_vocabulary(output_dir)


    return score


def predict(dataloader, model, proba=True, progress_bar=True):
    logits_complete = []
    # Evaluate data for one epoch
    if progress_bar:
        loader = tqdm(dataloader)
    else:
        loader = dataloader
    for batch in loader:
        
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        if len(batch)==3:
            b_input_ids, b_input_mask, _ = batch
        else:
            b_input_ids, b_input_mask = batch


        # Telling the model not to compute or store gradients, saving memory andspeeding up validation
        with torch.no_grad():        
            outputs = model(b_input_ids, 
                            token_type_ids=None,  # not a 2-sentence task
                            attention_mask=b_input_mask)

        # Get the "logits" output by the model. The "logits" are the output
        # values prior to applying an activation function like the softmax.
        logits = outputs[0]
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        logits_complete.append(logits)
        
        del outputs        
        torch.cuda.empty_cache() #release GPU memory
        
        pred = np.concatenate(logits_complete) #flatten batches
    if proba:
        return softmax(pred, axis=1)
    else:
        return pred
    
def predict_with_model(dataloader, model_path, proba=True):
    model = CamembertForSequenceClassification.from_pretrained(model_path)
    if torch.cuda.is_available():
        model.cuda()
    return predict(dataloader, model, proba)
