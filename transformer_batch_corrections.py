import re
import torch
import math
import random

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import train_test_split

from transformers import TransformNet
from asses_batch_effect import batchless_entropy_estimate, batchless_entropy_distribuions, fisher_kldiv, abs_effect_estimate, compute_F_stat
from report_on_correction import make_report, correction_scatter, batch_density_plot


class Correction_data(nn.Module):
    def __init__(self, CrossTab, depth, correction_reg, batch_reg, n_batches, batch_size, test_size, random_state, n_divisions, n_overlap,
                 minibatch_size = 200, number_bench = 10, train_on_all = False):
      
        super().__init__()

        self.CrossTab = CrossTab
        self.resampled = False
        self.corrected_data = CrossTab
        self.number_bench = number_bench
        self.random_state = random_state
        self.n_overlap = n_overlap
        self.n_divisions = n_divisions
        self.finetune_training = False
        
        ## Data embedding
        self.TRAIN_DATA, self.TEST_DATA, self.FULL_DATA, self.METADATA = make_dataset_transformer(CrossTab = CrossTab, 
                                                                                                  emb = 6, 
                                                                                                  n_batches = n_batches,
                                                                                                  test_size = test_size, 
                                                                                                  random_state = random_state)

        if (train_on_all):
            print("Will train on the entire dataset.\n")
            self.TRAIN_DATA = self.FULL_DATA
        
        self.trainloader = torch.utils.data.DataLoader(self.TRAIN_DATA, shuffle = True, 
                                                       batch_size = minibatch_size)
        self.testloader = torch.utils.data.DataLoader(self.TEST_DATA, shuffle = False, 
                                                      batch_size = minibatch_size)
        self.loader = torch.utils.data.DataLoader(self.FULL_DATA, shuffle = False, 
                                                  batch_size = minibatch_size)
        self.finetune_loaders = []
        
        # shuffled_dataset = torch.utils.data.Subset(my_dataset, torch.randperm(len(my_dataset)).tolist())
        # dataloader = DataLoader(shuffled_dataset, batch_size=4, num_workers=4, shuffled=False)

        ## Normalizing reg factor using mean magnitude of the batch means
        self.original_batch_means = torch.tensor(CrossTab.values).view([len(torch.tensor(CrossTab.values)), n_batches, batch_size])
        self.original_batch_means = torch.mean(self.original_batch_means, 2)
        self.original_batch_means = float(torch.mean(torch.abs(self.original_batch_means)))
                                            

        ## Important self variables
        self.minibatch_size = minibatch_size
        self.n_minibatch = math.ceil(len(self.TRAIN_DATA)/minibatch_size)
        self.correction_reg = correction_reg
        self.batch_reg = batch_reg
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.train_n = len(self.TRAIN_DATA)
        self.test_n = len(self.TEST_DATA)
        self.data_n = len(self.FULL_DATA)
        self.batchless_entropy, self.batchless_entropy_std = batchless_entropy_estimate(n_batches = self.n_batches,
                                                                                        batch_size = self.batch_size)
        self.batchless_entropy_distributions = []
        
        ## The network
        self.network = TransformNet(emb = self.batch_size, seq_length = self.n_batches, depth = depth,
                                    n_batches = self.n_batches, batch_size = self.batch_size)
        
        ## The optimizers
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr = 1e-4, betas = (0.9, 0.999))
        # self.optimizer = torch.optim.Adam(self.network.parameters(), lr = 1e-5, betas = (0.9, 0.999))

        ## Set the weights of the final layer to zero. This is so that that the inital corrections are all zero.
        self.network.correction[2].weight.data.fill_(0)
        self.network.correction[2].bias.data.fill_(0)

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.network = self.network.to(self.device)
        # self.make_metrics()

    def resampled_training(self, megabatch_size):
        self.resampled = True
        CrossTab_train = self.CrossTab.iloc[self.METADATA['train_idx']]
        CrossTab_test = self.CrossTab.iloc[self.METADATA['test_idx']]
        resampled_data = make_resampled_dataset(CrossTab = CrossTab_train, n_batches = self.n_batches, 
                                                minibatch_size = self.minibatch_size)
        self.resampled_trainloader = torch.utils.data.DataLoader(resampled_data, shuffle = True, batch_size = megabatch_size)
        resampled_data = make_resampled_dataset(CrossTab = CrossTab_test, n_batches = self.n_batches, 
                                                minibatch_size = self.minibatch_size)
        self.resampled_testloader = torch.utils.data.DataLoader(resampled_data, shuffle = False, batch_size = megabatch_size)
        self.train_n = len(self.resampled_trainloader)
        self.test_n = len(self.resampled_testloader)

    def make_finetune_loaders(self, megabatch_size):
        self.batchless_entropy_distributions = []
        self.finetune_loaders = []  
        loader = torch.utils.data.DataLoader(self.TRAIN_DATA, shuffle = False, batch_size = megabatch_size)
        div_size = len(self.TRAIN_DATA)//self.n_divisions
        step_size = div_size//self.n_overlap
        windows = []
        for j in range(self.n_overlap):
            windows = windows + [(i*div_size + j*step_size, (i + 1)*div_size + j*step_size) for i in range(self.n_divisions)]

        tensor_crosstab = []
        for _, _, y, mask in loader:
            y, z = self.compute_correction(y, mask)
            tensor_crosstab.append(y-z)

        tensor_crosstab = torch.cat(tensor_crosstab)
        F_stat = compute_F_stat(tensor_crosstab, self.n_batches, self.batch_size)
        F_df = pd.DataFrame({'index' : range(0, len(F_stat)), 'F_stat' : F_stat.cpu().detach().numpy()})

        A = F_df.sort_values(by = 'F_stat', ascending = False)
        A = pd.concat([A,A])

        for index, window in enumerate(windows):
            windows[index] = A.iloc[window[0]:window[1]]['index'].to_list()

        for index, section in enumerate(windows):
            section = self.CrossTab.iloc[self.METADATA['train_idx']].iloc[section]
            resampled_data = make_resampled_dataset(CrossTab = section, n_batches = self.n_batches, 
                                                    minibatch_size = self.minibatch_size)
            resampled_loader = torch.utils.data.DataLoader(resampled_data, shuffle = True, batch_size = megabatch_size)
            self.finetune_loaders = self.finetune_loaders + [resampled_loader]
        self.batchless_entropy_distributions = batchless_entropy_distribuions(self.n_batches, self.batch_size, self.n_divisions, self.n_overlap)

        # div_size = len(self.loader.dataset)//n_div
        # tensor_crosstab = []
        # for _, _, y, mask in self.loader:
        #     y, z = self.compute_correction(y, mask)
        #     tensor_crosstab.append(y-z)

        # tensor_crosstab = torch.cat(tensor_crosstab)
        # F_stat = compute_F_stat(tensor_crosstab, self.n_batches, self.batch_size)
        # F_df = pd.DataFrame({'index' : range(0, len(F_stat)), 'F_stat' : F_stat.cpu().detach().numpy()})

        # C = F_df.sort_values(by = 'F_stat', ascending = False)

        

    def make_metrics(self):
        self.robust_loaders = []
        for i in range(0, self.number_bench):
            # new_order = list(range(0, len(self.testloader.dataset)))
            # new_order = random.sample(new_order, k = len(new_order))
            # print(new_order[0:10])
            # shuffled_dataset = Subset(self.testloader.dataset, indices = new_order)
            # self.robust_loaders += [torch.utils.data.DataLoader(shuffled_dataset, shuffle = False, 
            #                                                     batch_size = self.minibatch_size)]
            self.robust_loaders += [self.shuffle_loader(self.testloader)]

    def compute_correction(self, y, mask):
        y, mask = y.clone().detach().to(self.device), mask.detach().to(self.device)
        reshape_dims = [y.size(i) for i in range(0, y.dim()-1)]
        x = y.reshape(*reshape_dims, self.n_batches, self.batch_size).float()
        z = self.network(x, mask)
        return y, z

    ## Distance based on batch effect present in data y.
    # def fisher_dist(self, y, z):
    #     batch_dist = fisher_kldiv(y-z, 
    #                               self.n_batches, 
    #                               self.batch_size, 
    #                               self.batchless_entropy)
    #     if batch_dist.dim() > 1:
    #         batch_dist = torch.mean(batch_dist, 1)
    #         batch_dist = batch_dist * np.sqrt(self.minibatch_size)
    #     return(batch_dist)

    def general_objective(self, y, z, ref_mean, ref_std):
        batch_dist = fisher_kldiv(y-z, 
                                  self.n_batches, 
                                  self.batch_size, 
                                  0)
        dist_mean = torch.mean(batch_dist)
        dist_std = torch.std(batch_dist, unbiased = True)

        gaussian_kl1 = torch.log(dist_std/ref_std) + (ref_std**2 + (dist_mean - ref_mean)**2)/(2*dist_std**2) - 1/2
        gaussian_kl2 = torch.log(ref_std/dist_std) + (dist_std**2 + (dist_mean - ref_mean)**2)/(2*ref_std**2) - 1/2
        return gaussian_kl1 + gaussian_kl2

    def objective(self, y, z):
        # batch_dist = self.fisher_dist(y, z)
        # dist_mean = torch.mean(batch_dist)
        # dist_std = torch.std(batch_dist, unbiased = True)
        # ref_std = self.batchless_entropy_std

        # gaussian_kl1 = torch.log(dist_std/ref_std) + (ref_std**2 + dist_mean**2)/(2*dist_std**2) - 1/2
        # gaussian_kl2 = torch.log(ref_std/dist_std) + (dist_std**2 + dist_mean**2)/(2*ref_std**2) - 1/2
        return self.general_objective(y, z, -self.batchless_entropy, self.batchless_entropy_std)
        # return gaussian_kl2

    def reg_objective(self, y, z):
        view_args = []
        y_dim = y.dim()
        for index in range(0, y_dim-1):
            view_args = view_args + [y.size(index)]
        view_args = view_args + [self.n_batches, self.batch_size]
        data_means = torch.mean(torch.mean((y-z).view(*view_args), y_dim).view(-1, self.n_batches), 0)
        batch_reg = self.batch_reg * torch.sum(abs(data_means))

        correction_reg = self.correction_reg * torch.mean(torch.abs(z)) / self.original_batch_means
        # reg_dist = reg_dist**2
        return correction_reg + batch_reg

    def finetune_objective(self, y, z, index):
        batchless_dist = self.batchless_entropy_distributions[index]
        return self.general_objective(y, z, -batchless_dist[0], batchless_dist[1])

    # def robust_metric(self):
    #     self.eval()
    #     abs_loss = 0
    #     for loader in self.robust_loaders:
    #         loader_loss = 0
    #         for _, _, y, mask in loader:
    #                 y, z = self.compute_correction(y, mask)
    #                 # loss = self.objective(y, z)
    #                 loss = self.fisher_dist(y, z)
    #                 loader_loss += abs(float(loss))
    #         abs_loss += loader_loss
    #     robust_estimate = abs_loss/(len(self.robust_loaders) * len(loader.dataset))
    #     return(robust_estimate)

    # def shuffle_loader(self, loader, A_prop = 0.65):
    #     B_prop = 1 - A_prop
    #     crosstab = []
    #     for _, _, y, mask in loader:
    #         y, z = self.compute_correction(y, mask)
    #         crosstab.append(y-z)

    #     crosstab = torch.cat(crosstab)
    #     individual_distance = fisher_kldiv(crosstab, self.n_batches, self.batch_size, self.batchless_entropy)
    #     individual_distance = pd.DataFrame({'index' : range(0, len(individual_distance)),
    #                                         'distance' : individual_distance.cpu().detach().numpy()})

    #     C = individual_distance.sort_values(by = 'distance', ascending = False)
    #     A = C['index'].to_list()[:len(C)//2]
    #     B = C['index'].to_list()[len(C)//2:]

    #     A = random.sample(A, k = len(A)) 
    #     B = random.sample(B, k = len(A))
    #     n_minibatch = len(loader)
    #     A_last = 0
    #     B_last = 0

    #     new_order = []
    #     for minibatch_i in range(0, n_minibatch):
    #         if (minibatch_i % 2 == 0):
    #             new_minibatch = A[A_last:A_last + math.floor(A_prop * self.minibatch_size)] + B[B_last:B_last + math.ceil(B_prop * self.minibatch_size)]
    #             A_last += math.floor(A_prop * self.minibatch_size)
    #             B_last += math.ceil(B_prop * self.minibatch_size)
    #             new_order += new_minibatch
    #         else:
    #             new_minibatch = A[A_last:A_last + math.ceil((1 - A_prop) * self.minibatch_size)] + B[B_last:B_last + math.floor((1 - B_prop) * self.minibatch_size)]
    #             A_last += math.ceil(B_prop * self.minibatch_size)
    #             B_last += math.floor(A_prop * self.minibatch_size)
    #             new_order += new_minibatch
        
    #     # print(new_order[0:10])
    #     dataset_shuffled = Subset(loader.dataset, indices = new_order)
    #     new_loader = torch.utils.data.DataLoader(dataset_shuffled, shuffle = True, batch_size = self.minibatch_size)
    #     return(new_loader)

    def train_model(self, epochs, abs_effect_cutoff, finetune_training = False, resample_training = None, minibatch_bias = None, report_frequency = 50, run_name = ""):
        train_complete = False
        train_loss_all, test_loss_all = [], []
        # abs_train_all = []
        # abs_test_all = []
        abs_all = []
        training_loss = 0
        total_reg_loss = 0

        for epoch in range(epochs):
            if ((epoch % report_frequency == 0) and not train_complete):
                self.eval()
                test_loss = 0
                # train_data_corrected = []
                # test_data_corrected = []
                data_corrected = []


                if self.resampled:
                    for y, mask in self.resampled_testloader:
                        y, z = self.compute_correction(y, mask)
                        raw_loss = self.objective(y, z)
                        test_loss += float(raw_loss)
                else:
                    for _, _, y, mask in self.testloader:
                        y, z = self.compute_correction(y, mask)
                        raw_loss = self.objective(y, z)
                        training_loss += float(raw_loss)

                for _, _, y, mask in self.loader:
                    y, z = self.compute_correction(y, mask)
                    data_corrected.append(y-z)
                
                # for _, _, y, mask in self.trainloader:
                #     y, z = self.compute_correction(y, mask)
                #     train_data_corrected.append(y-z)

                test_loss = test_loss / self.test_n
                # full_loss = full_loss / self.data_n
                test_loss_all.append(test_loss)
                # full_loss_all.append(full_loss)

                data_corrected = torch.cat(data_corrected)
                abs_effect = float(abs_effect_estimate(data_corrected, self.n_batches, self.batch_size, self.batchless_entropy))
                # train_data_corrected = torch.cat(train_data_corrected)
                # abs_effect_train = float(abs_effect_estimate(train_data_corrected, self.n_batches, self.batch_size, self.batchless_entropy))

                abs_all.append(abs_effect)
                # abs_train_all.append(abs_effect_train)
                data_corrected = data_corrected.cpu().detach().numpy()
                data_corrected = pd.DataFrame(data_corrected)
                data_corrected.index = self.CrossTab.index
                column_mapping = dict(zip(data_corrected.columns, self.CrossTab.columns))
                data_corrected = data_corrected.rename(columns = column_mapping)
                self.corrected_data = data_corrected

                
                # robust_stop_metric = self.robust_metric()
                make_report(data_corrected, n_batches = self.n_batches, batch_size = self.batch_size,  
                            train_idx = self.METADATA['train_idx'], test_idx = self.METADATA['test_idx'],
                            prefix = run_name, suffix = "_epoch_" + format(epoch) + "_abs_" + format(round(abs_effect, 5)) + 
                                                        "_train_" + format(round(training_loss, 5)) + "_test_" + format(round(test_loss, 5)))
                print("Epoch " + format(epoch) + " report : testing loss is " + format(test_loss) + 
                      " while train loss is " + format(training_loss) + " abs effect in test data is " + format(abs_effect) + 
                      " reg loss in training is " + format(total_reg_loss) + "\n")

                
                if abs_effect < abs_effect_cutoff:
                    train_complete = True

                if minibatch_bias is not None:
                    self.trainloader = self.shuffle_loader(self.trainloader, minibatch_bias)

                if resample_training is not None:
                    self.resampled_training(resample_training)

                if finetune_training:
                    self.make_finetune_loaders(resample_training)
                        

                # if (robust_stop_metric < robust_cutoff):
                #     train_complete = True

            if (not train_complete):
                self.train()
                training_loss = 0
                ## The training is done here.
                if finetune_training:
                    for index in range(len(self.finetune_loaders[0])):
                        raw_loss = 0
                        reg_loss = 0
                        for loader_index, finetune_loader in enumerate(self.finetune_loaders):
                            y, mask = next(iter(finetune_loader))
                            self.optimizer.zero_grad()
                            y, z = self.compute_correction(y, mask)
                            raw_loss += self.finetune_objective(y, z, loader_index)
                            reg_loss += self.reg_objective(y, z)
                        loss = raw_loss + reg_loss
                        loss.backward()
                        self.optimizer.step()
                        training_loss += float(raw_loss)
                        total_reg_loss += float(reg_loss)
                        # for y, mask in finetune_loader:
                            # self.optimizer.zero_grad()
                            # y, z = self.compute_correction(y, mask)
                            # raw_loss = self.finetune_objective(y, z, index)
                            # loss = raw_loss + self.reg_objective(z)
                            # loss.backward()
                            # self.optimizer.step()
                            # training_loss += float(raw_loss)
                elif self.resampled:
                    for y, mask in self.resampled_trainloader:
                        self.optimizer.zero_grad()
                        y, z = self.compute_correction(y, mask)
                        raw_loss = self.objective(y, z)
                        loss = raw_loss + self.reg_objective(y, z)
                        loss.backward()
                        self.optimizer.step()
                        training_loss += float(raw_loss)
                else:
                    for _, _, y, mask in self.trainloader:
                        self.optimizer.zero_grad()
                        y, z = self.compute_correction(y, mask)
                        raw_loss = self.objective(y, z)
                        loss = raw_loss + self.reg_objective(y, z)
                        loss.backward()
                        self.optimizer.step()
                        training_loss += float(raw_loss)

                training_loss = training_loss / self.train_n
                train_loss_all.append(training_loss)
                print("Training loss is " + format(training_loss))

            if (epoch % report_frequency == 0 and epoch > 0 and not train_complete):
                ## We also make a plot of the training, testing and full losses during training.
                fig, plots = plt.subplots(1, 2, figsize = (10,5))

                plot_index = [j * report_frequency for j in range(len(test_loss_all))]
                plots[0].plot(train_loss_all, label = 'Train loss')
                plots[0].plot(plot_index, test_loss_all, label = 'Test loss')
                plots[0].legend()
                plots[0].set_title("Network loss")
                plot_index = [j * report_frequency for j in range(len(abs_all))]
                plots[1].plot(plot_index, abs_all, label = "Absolute batch effect in data")
                # plots[1].plot(plot_index, abs_test_all, label = "Test abs effect")
                plots[1].legend()
                plots[1].set_title("Absolute effect estimate")
                
                plot_title = "All losses epochs " + format(epoch)
                fig.suptitle(plot_title)
                path = "./loss_summaries/" + run_name + "_" + plot_title + ".png"
                fig.savefig(path)
                fig.clf()
                plt.close(fig)

        ## Finished loop, saving corrected data
        data_corrected_output = []
        self.eval()
        for _, _, y, mask in self.loader:
            y, z = self.compute_correction(y, mask)

            data_corrected_output.append(y-z)

        ## Place corrected data into data frame
        data_corrected_output = torch.cat(data_corrected_output).cpu().detach().numpy()
        data_corrected_output = pd.DataFrame(data_corrected_output)
        data_corrected_output.index = self.CrossTab.index
        column_mapping = dict(zip(data_corrected_output.columns, self.CrossTab.columns))
        data_corrected_output = data_corrected_output.rename(columns = column_mapping)
        self.corrected_data = data_corrected_output

    
    def scatter_comparison(self, alpha = 0.07):
        correction_scatter(original_data = self.CrossTab, 
                           corrected_data = self.corrected_data, 
                           n_batches = self.n_batches, 
                           batch_size = self.batch_size,
                           alpha = alpha)


    def batch_density_plot(self, *args, corrected = False):
        if (corrected):
            plot_title = "Corrected"
            data = self.corrected_data
        else:
            plot_title = "Original"
            data = self.CrossTab
        plot_title = plot_title + " batch means"
        
        batch_density_plot(data, self.n_batches, self.batch_size, 
                           plot_title, *args)

                          





## Helper function for making the dataset used by the networks. Made from CrossTab coming out of R.
## The rows of the pandas dataset x represent peptides, and columns represent samples.
def make_dataset_transformer(CrossTab, emb, n_batches, random_state, test_size = 0.20, start_char = "!", padding = "$"):
    ## The "^" character is needed for the amino acids to be extracted from each string properly.
    x = CrossTab
    pattern_dict = "[^a-z*$A-Z0-9!;_-]*"

    ## Helper function to encode position
    def positional_encoding(emb, seq_length, base = 1000):
        encoding = []
        for pos in range(seq_length):
            pos_enc = []
            N = int(emb/2)
            for i in range(N):
                pos_enc = np.append(pos_enc, [math.cos(pos / (base ** (2*i/emb))),
                                              math.sin(pos / (base ** (2*i/emb)))])
            encoding.append(pos_enc)

        encoding = torch.tensor(np.array(encoding))
        return(encoding)


    ## Helper function for splitting feature names into characters
    def split_helper(feature_name):
        return list(filter(None, re.split(pattern_dict, feature_name)))

        
    ## Helper function for masking the appended padding tokens '$'.
    def mask_helper(seq_length, max_pep_length):
        mask = []
        for i in range(max_pep_length):
            new_row = [float(0)] * seq_length + [float('inf')] * (max_pep_length - seq_length)
            mask.append(new_row)
        return(mask)
            
    x.columns = x.columns.astype(str)
    x = x.dropna() # TODO: warning message here

    feature_names_og = x.index
    max_pep_length = max(feature_names_og.map(len))
    ## Adding padding to make all peptide sequences the same length
    feature_names = [start_char + feature_name + padding*(max_pep_length - len(feature_name)) 
                                                        for feature_name in feature_names_og]
    ## Have to refresh the value, as we put a start_char above
    max_pep_length += 1

    masks_peptide = []
    masks_data = []
    ## Have to add 1 because we placed a special token "!" to the beginning of each peptide
    for feature_name in feature_names_og:
      masks_peptide.append(mask_helper(len(feature_name) + 1, max_pep_length))
      masks_data.append(mask_helper(n_batches, n_batches))
    masks_peptide = torch.tensor(masks_peptide)
    masks_data = torch.tensor(masks_data)
    sample_names  = x.columns
    n_features   = len(x) 

    symbols = ''

    for i in range(n_features): 
      symbols = symbols + feature_names[i]

    symbols = ''.join(set(symbols))
    symbols = list(filter(None, re.split(pattern_dict, symbols)))
    symbols_dict = {}

    j = 0
    ## This dictionary will translate the symbols to integers.
    for xx in symbols:
      symbols_dict[xx] = j
      j += 1

    n_letters = len(symbols_dict)

    feature_names = list(map(split_helper, feature_names))
    feature_names = [[symbols_dict[key] for key in feature_name] for feature_name in feature_names]   
    feature_names = torch.tensor(feature_names) 
    positions = positional_encoding(emb, max_pep_length)
    positions = positions.repeat(n_features, 1, 1).float()

    embedding = torch.nn.Embedding(n_letters, emb)
    embedded = embedding(feature_names) + positions

    x = torch.tensor(x.values)
    dataset = TensorDataset(embedded, masks_peptide, x, masks_data) # convert to tensor
    train_idx, test_idx = train_test_split(range(n_features), # make indices
                                          test_size = test_size,
                                          random_state = random_state)

    train_dataset = Subset(dataset, train_idx) # generate subset based on indices
    test_dataset  = Subset(dataset, test_idx)

    metadata = {
      'feature_names_og' : feature_names_og,
      'feature_names'    : feature_names,
      'positions'        : positions,
      'sample_names'     : sample_names,
      'n_features'       : n_features,
      'max_pep_len'      : max_pep_length, 
      'train_idx'        : train_idx,
      'test_idx'         : test_idx,
      'symbols_dict'     : symbols_dict,
      'symbol_embedding' : embedding
      }
      
    return train_dataset, test_dataset, dataset, metadata


def make_resampled_dataset(CrossTab, n_batches, minibatch_size, n_stack = 4):
    # train_idx, test_idx = train_test_split(range(len(CrossTab)), # make indices
    #                                     test_size = test_size,
    #                                     random_state = random_state)
    CrossTab.columns = CrossTab.columns.astype(str)
    CrossTab = CrossTab.dropna() # TODO: warning message here

    # train = CrossTab.iloc[train_idx]
    # test = CrossTab.iloc[test_idx]
    train = CrossTab

    train = pd.concat([train] * n_stack)
    # test = pd.concat([test] * n_stack)
    train = train.sample(frac = 1)
    # test = test.sample(frac = 1)

    ## Helper function for masking the appended padding tokens '$'.
    def mask_helper(seq_length, max_pep_length):
        mask = []
        for i in range(max_pep_length):
            new_row = [float(0)] * seq_length + [float('inf')] * (max_pep_length - seq_length)
            mask.append(new_row)
        return(mask)

    masks_train = []
    # masks_test = []
    for feature_name in train.index:
        masks_train.append(mask_helper(n_batches, n_batches))
    # for feature_name in test.index:
    #     masks_test.append(mask_helper(n_batches, n_batches))

    sample_names  = train.columns
    train = torch.tensor(train.values)
    # test = torch.tensor(test.values)
    masks_train = torch.tensor(masks_train)
    # masks_test = torch.tensor(masks_test)

    ## Stack the minibatches. Dropping the last one to keep dimensions exact.
    train = torch.stack(list(torch.split(train, minibatch_size))[:-1])
    # test = torch.stack(list(torch.split(test, minibatch_size))[:-1])
    masks_train = torch.stack(list(torch.split(masks_train, minibatch_size))[:-1])
    # masks_test = torch.stack(list(torch.split(masks_test, minibatch_size))[:-1])

    train_dataset = TensorDataset(train, masks_train)
    # test_dataset = TensorDataset(test, masks_test)

    return train_dataset

