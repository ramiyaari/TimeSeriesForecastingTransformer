import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

import torch
from torch.utils.data import DataLoader, TensorDataset

import copy 
import sys


quantiles = [0.010, 0.025, 0.050, 0.100, 0.150, 0.200, 0.250, 0.300, 0.350,
             0.400, 0.450, 0.500, 0.550, 0.600, 0.650, 0.700, 0.750, 0.800,
             0.850, 0.900, 0.950, 0.975, 0.990]
nquantiles = len(quantiles)
q_median_ind = quantiles.index(0.5)

opt_target = 'q-loss' #'WIS' #

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Using CPU")

def load_ili_data(state):
    if(state=='US'):
        ILI_df = pd.read_csv('./data/ILI_national_2002_2024.csv')
        ILI_df = ILI_df[['date','year','week','weighted_ili']]
    else:
        ILI_df = pd.read_csv('./data/ILI_states_2010_2024.csv')
        ILI_df = ILI_df[['date','year','week',state]]
        ILI_df = ILI_df.rename(columns={state:'weighted_ili'})

    # ILI_df = ILI_df[ILI_df.week!=53] #ignore week 53 for now
    ILI_df['date'] = pd.to_datetime(ILI_df['date'])
    ILI_df = ILI_df[(ILI_df.date<'2020-06-28') | (ILI_df.date>='2022-07-01')] #remove covid years
    ILI_df = ILI_df.set_index('date')
    return ILI_df

# def quantile_loss(q, y, f):
#     # q: quantile, y: true values, f: predicted quantiles
#     e = (y - f)
#     loss = torch.max(q * e, (q - 1) * e)
#     return loss.mean()

def quantile_loss(q: float, y, f):
    """
    Calculate the quantile loss using either NumPy arrays or PyTorch tensors.
    
    Args:
    q (float): Quantile to be estimated, should be between 0 and 1.
    y (np.ndarray or torch.Tensor): True values.
    f (np.ndarray or torch.Tensor): Predicted quantiles.
    
    Returns:
    Quantile loss.
    """
    if isinstance(y, np.ndarray) and isinstance(f, np.ndarray):
        e = y - f
        loss = np.maximum(q * e, (q - 1) * e) #np.where(e >= 0, q * e, (q - 1) * e)
        return loss.mean()
    elif isinstance(y, torch.Tensor) and isinstance(f, torch.Tensor):
        e = y - f
        loss = torch.max(q * e, (q - 1) * e) #torch.where(e >= 0, q * e, (q - 1) * e)
        return loss.mean()
    else:
        raise TypeError("Inputs must be either both NumPy arrays or both PyTorch tensors.")


def wis_loss(q, y, f):
    # Generate quantile pairs
    n = len(q)
    quantile_pairs_ind = [(i, n-i-1) for i in range(n//2)]

    # Calculate Interval Scores (IS) for specified pairs of quantiles
    wis = 0
    for (q_lower_ind, q_upper_ind) in quantile_pairs_ind:
        p = q[q_upper_ind] - q[q_lower_ind]
        alpha = 1 - p

        # Retrieve predictions for the upper and lower quantiles
        L = f[:,:,q_lower_ind]
        U = f[:,:,q_upper_ind]

        # Interval Score calculation
        IS = (U - L) + (2 / alpha) * ((L - y) * (y < L) + (y - U) * (y > U))
        
        # Weight for each interval score, using alpha/2 as described
        wis += (alpha / 2) * IS.mean()  # mean of IS over all observations

    # Evaluate median accuracy separately if it is a distinct quantile
    if 0.5 in q:
        median_predictions = f[:,:,q.index(0.5)]
        median_error = abs(median_predictions - y).mean()
        wis += median_error

    K = len(quantile_pairs_ind)
    wis = wis/(K+0.5)
    return wis


def get_data_loader(ts_data, pred_list, input_length, output_length, 
                    weeks_ahead, batch_size, woy=None, add_ts_features=False,
                    shuffle=True, drop_last=True):
    
    SOS = np.float32(-2) #0

    pred_list2 = None
    if(pred_list is not None):
        if(len(pred_list)!=weeks_ahead-1):
            sys.exit("get_data_loader: len(pred_list)!=weeks_ahead-1")
        pred_list2 = []
        for j, pred in enumerate(pred_list):
            pred_list2.append(np.expand_dims(np.append(np.repeat(np.nan, len(ts_data)-len(pred)),pred[:,q_median_ind]),-1))
        
    ix = range(1, len(ts_data) - input_length - output_length - weeks_ahead + 2)

    if(woy is not None):
        # Cyclical encoding of week of the year
        woy_sin = np.sin(2 * np.pi * woy / 52)
        woy_cos = np.cos(2 * np.pi * woy / 52)


    inputs = []
    targets = []

    for i in ix:
        input_sequence = ts_data[i:i+input_length]
        input_diff = np.diff(ts_data[i-1:i+input_length].squeeze()).reshape(-1,1)
        if pred_list2 is not None:
            input_sequence0 = input_sequence
            input_sequence = input_sequence[(weeks_ahead-1):]
            for j, pred in enumerate(pred_list2):
                input_sequence = np.append(input_sequence,pred[(i+input_length+j)]).reshape(-1,1)
            input_diff = np.diff(np.insert(input_sequence,0,input_sequence0[weeks_ahead-2]).squeeze()).reshape(-1,1)
        if woy is not None:
            if pred_list2 is not None:
                input_woy_sin = woy_sin[i+weeks_ahead-1:i+input_length+weeks_ahead-1]
                input_woy_cos = woy_cos[i+weeks_ahead-1:i+input_length+weeks_ahead-1]
            else:    
                input_woy_sin = woy_sin[i:i+input_length]
                input_woy_cos = woy_cos[i:i+input_length]
            input_sequence = np.stack([input_sequence, input_woy_sin, input_woy_cos], axis=1).squeeze()  # Shape will be [input_length, 3]
        if(add_ts_features):
            # currently adding onty diff ts
            input_sequence = np.hstack([input_sequence, input_diff]).squeeze()  # Shape will be [input_length, 3]

        target_sequence = np.insert(ts_data[i+input_length+weeks_ahead-1:i+input_length+output_length+weeks_ahead-1],0,SOS)
        inputs.append(torch.from_numpy(input_sequence.astype(np.float32)))
        targets.append(torch.from_numpy(target_sequence.astype(np.float32)))

    # Convert lists to tensors
    inputs = torch.stack(inputs)
    targets = torch.stack(targets)

    # Create dataset and dataloader
    dataset = TensorDataset(inputs, targets)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return (data_loader)



def run_training_loop(num_epochs, optimizer, model, min_loss, training_loss, training_loader):

    model.train()
    best_model = copy.deepcopy(model) 
    cur_epoch = len(training_loss)
    for epoch in range(cur_epoch,cur_epoch+num_epochs):

        avg_train_loss = 0
        for input, target in training_loader:
            input, target = input.to(device), target.to(device)
            tgt_input = target[:, :-1] 
            tgt_output = target[:, 1:] 
            output = model(input, tgt_input) 
            optimizer.zero_grad()
            if(opt_target=='q-loss'):
                # loss = sum(quantile_loss(q, tgt_output.view(-1), output[:,:,i].view(-1))
                loss = sum(quantile_loss(q, tgt_output.reshape(-1), output[:,:,i].reshape(-1))
                                for i, q in enumerate(quantiles))    
            else:
                loss = wis_loss(quantiles, tgt_output, output)  
            loss.backward()
            optimizer.step()
            avg_train_loss += loss.item() * input.size(0)
        avg_train_loss /= len(training_loader.dataset)
        
        training_loss.append(avg_train_loss)
        if avg_train_loss < min_loss:
            best_model = copy.deepcopy(model) 
            min_loss = avg_train_loss
        print(f'Epoch {epoch}: training loss: {round(avg_train_loss, 4)}')
        
    return (best_model, optimizer, min_loss, training_loss)


def run_training_loop_with_validation(num_epochs, optimizer, model, min_loss, 
                                      training_loss, validation_loss, 
                                      training_loader, val_loader):

    # best_model = copy.deepcopy(model) 
    best_model_state = model.state_dict()
    cur_epoch = len(training_loss)
    for epoch in range(cur_epoch,cur_epoch+num_epochs):
        
        model.train()
        avg_train_loss = 0
        for input, target in training_loader:
            input, target = input.to(device), target.to(device)
            tgt_input = target[:, :-1] 
            tgt_output = target[:, 1:] 
            output = model(input, tgt_input) 
            optimizer.zero_grad()
            if(opt_target=='q-loss'):
                loss = sum(quantile_loss(q, tgt_output.reshape(-1), output[:,:,i].reshape(-1))
                                for i, q in enumerate(quantiles))    
            else:
                loss = wis_loss(quantiles, tgt_output, output)  
            loss.backward()
            optimizer.step()
            avg_train_loss += loss.item() * input.size(0)
        avg_train_loss /= len(training_loader.dataset)
        training_loss.append(avg_train_loss)

        model.eval()
        with torch.no_grad():
            avg_val_loss = 0
            for input, target in val_loader:
                input, target = input.to(device), target.to(device)
                tgt_input = target[:, :-1] 
                tgt_output = target[:, 1:] 
                output = model(input, tgt_input) 
                if(opt_target=='q-loss'):
                    val_loss = sum(quantile_loss(q, tgt_output.reshape(-1), output[:,:,i].reshape(-1))
                                for i, q in enumerate(quantiles))    
                else:
                    val_loss = wis_loss(quantiles, tgt_output, output)  
                avg_val_loss += val_loss.item() * input.size(0)
            avg_val_loss /= len(val_loader.dataset)
            validation_loss.append(avg_val_loss)
            if avg_val_loss < min_loss:
                # best_model = copy.deepcopy(model) 
                best_model_state = model.state_dict()
                min_loss = avg_val_loss

        print(f'epoch {epoch}: training loss: {round(avg_train_loss, 4)}, validation loss: {round(avg_val_loss,4)}')

    model.load_state_dict(best_model_state)    
    return (model, optimizer, min_loss, training_loss, validation_loss)
    # return (best_model, optimizer, min_loss, training_loss, validation_loss)



def get_model_pred(data, prev_pred, model, weeks_ahead, input_length, 
                   output_length, batch_size, woy=None, add_ts_features=False):
    
    data_loader = get_data_loader(data, prev_pred, 
                                  input_length, output_length, 
                                  weeks_ahead, batch_size, woy, add_ts_features,
                                  shuffle=False, drop_last=False)
    pred = []
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # No gradients needed for validation, reduces memory and computation
        for input, target in data_loader:
            input, target = input.to(device), target.to(device)
            tgt_input = target[:, :-1] 
            output = model(input, tgt_input) 
            out = output[:,0,:].to('cpu').detach().numpy().squeeze()
            if(len(out.shape)==1):
                out = np.expand_dims(out,0) #last batch may contain only one input
            pred.append(out)

    pred = np.concatenate(pred, axis=0)
    return pred


def plot_pred_fit(pred_ili, ili, dates, weeks_ahead, state, use_dates_index=True):
    wis_score = np.round(wis_loss(quantiles,ili,np.expand_dims(pred_ili,1)),3)
    q_loss = np.round(sum(quantile_loss(q, ili.reshape(-1), 
                                         pred_ili[:,i].reshape(-1)) 
                                         for i, q in enumerate(quantiles)),3)
    print(f'weeks_ahead={weeks_ahead} (length={len(ili)}): WIS={wis_score}, Quantile loss={q_loss}')

    if(use_dates_index==False):
        dates = range(0,len(dates))

    lowq = 0.025
    uppq = 0.975
    q_low_ind = quantiles.index(lowq) #quantiles.index(0.25) #
    q_high_ind = quantiles.index(uppq) #quantiles.index(0.75) #
    plt.figure(figsize=(10, 4))
    plt.plot(dates,ili,'o--',markersize=3,label='data',color='black',alpha=0.75)
    plt.plot(dates,pred_ili[:,q_median_ind], label='pred median',alpha=0.75, color='green')
    plt.plot(dates,pred_ili[:,q_low_ind], label='pred low ({})'.format(lowq),alpha=0.75, color='blue')
    plt.plot(dates,pred_ili[:,q_high_ind], label='pred high ({})'.format(uppq),alpha=0.75, color='red')
    plt.xlabel('week')
    plt.ylabel('ILI')
    plt.title('state={}, horizon={} weeks (WIS={}, Q-loss={})'.format(state, weeks_ahead,wis_score,q_loss))
    plt.legend(loc=0)