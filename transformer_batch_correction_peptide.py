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
from asses_batch_effect import batchless_entropy_estimate, fisher_kldiv, fisher_kldiv_detailed, abs_effect_estimate
from report_on_correction import make_report, correction_scatter, batch_density_plot




class Correction_peptide(nn.Module):
    def __init__(self, CrossTab, emb, depth, n_batches, batch_size, test_size, minibatch_size, 
                 random_state, reg_factor = 0, heads = 5, ff_mult = 5):
      
        super().__init__()

        self.CrossTab = CrossTab
        self.corrected_data = CrossTab
        
        ## Data embedding
        self.TRAIN_DATA, self.TEST_DATA, self.FULL_DATA, self.METADATA = make_dataset_transformer(CrossTab = CrossTab, 
                                                                                                  emb = emb, 
                                                                                                  n_batches = n_batches,
                                                                                                  test_size = test_size, 
                                                                                                  random_state = random_state)
        
        self.trainloader = torch.utils.data.DataLoader(self.TRAIN_DATA, shuffle = True, 
                                                       batch_size = minibatch_size)
        self.testloader = torch.utils.data.DataLoader(self.TEST_DATA, shuffle = False, 
                                                      batch_size = minibatch_size)
        self.loader = torch.utils.data.DataLoader(self.FULL_DATA, shuffle = False, 
                                                  batch_size = minibatch_size)

        ## Important self variables
        self.seq_length = self.METADATA['max_pep_len']
        self.reg_factor = reg_factor
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.test_n = len(self.TEST_DATA)
        self.train_n = len(self.TRAIN_DATA)
        self.data_n = len(self.FULL_DATA)
        self.batchless_entropy = batchless_entropy_estimate(n_batches = self.n_batches,
                                                    batch_size = self.batch_size)
        self.individual_distance = fisher_kldiv_detailed(self.corrected_data, self.n_batches, self.batch_size, self.batchless_entropy)

        ## The network
        self.network = TransformNet(emb = emb, seq_length = self.seq_length, depth = depth, n_batches = n_batches, 
                                    batch_size = batch_size, heads = heads, ff_mult = ff_mult)
        
        ## The optimizers
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr = 1e-4, betas = (0.9, 0.999))

        ## Set the weights of the final layer to zero. This is so that that the inital corrections are all zero.
        self.network.correction[2].weight.data.fill_(0)
        self.network.correction[2].bias.data.fill_(0)

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.network = self.network.to(self.device)


    ## Distance based on batch effect present in data y.
    def objective_kldiv(self, y, z):
        loss_kl = fisher_kldiv(y-z, 
                               self.n_batches, 
                               self.batch_size, 
                               self.batchless_entropy)
        loss_kl = torch.abs(loss_kl)
        
        return loss_kl + self.reg_factor * torch.sum(z**2)


    ## Computes mse. y should be 'data - prediction'.
    def objective_mse(self, y, z):
        y = y - z
        return torch.sum(y**2) / (self.n_batches * self.batch_size)


    def train_model(self, epochs, loss_cutoff = 0, report_frequency = 10, early_stopping = 100, objective = "mse", run_name = ""):
        early_stopping_N = early_stopping // report_frequency
        train_complete = False
        train_loss_all = []
        test_loss_all = []
        full_loss_all = []

        if (objective == "batch_correction"):
            objective = self.objective_kldiv
        elif (objective == "mse"):
            objective  = self.objective_mse
        else:
            print("Must input a valid objective")

        for epoch in range(epochs):
            if ((epoch % report_frequency == 0) and not train_complete):
                self.eval()
                test_loss, training_loss, full_loss = 0, 0, 0
                data_corrected = []
                p_values = []
                
                for x, mask, y, _ in self.testloader:
                    x, mask = x.clone().detach().to(self.device), mask.detach().to(self.device) 
                    y, z = y.clone().detach().to(self.device), self.network(x, mask)
                    loss = objective(y, z)
                    test_loss += float(loss)

                for x, mask, y, _ in self.loader:
                    x, mask = x.clone().detach().to(self.device), mask.detach().to(self.device) 
                    y, z = y.clone().detach().to(self.device), self.network(x, mask)
                    loss = objective(y, z)
                    data_corrected.append((y-z).detach().cpu())
                    full_loss += float(loss)

                for x, mask, y, _ in self.trainloader:
                    x, mask = x.clone().detach().to(self.device), mask.detach().to(self.device) 
                    y, z = y.clone().detach().to(self.device), self.network(x, mask)
                    loss = objective(y, z)
                    training_loss += float(loss)


                test_loss = test_loss / self.test_n
                full_loss = full_loss / (self.test_n + self.train_n)
                test_loss_all.append(test_loss)
                full_loss_all.append(full_loss)
                data_corrected = torch.cat(data_corrected).cpu().detach().numpy()
                data_corrected = pd.DataFrame(data_corrected)
                
                make_report(data_corrected, n_batches = self.n_batches, batch_size = self.batch_size, 
                            prefix = run_name + "all_data_", suffix = format(epoch))
                print("Epoch " + format(epoch) + " report : testing loss is " + format(test_loss) + 
                      " while full loss is " + format(full_loss) + "\n")

                if (full_loss < loss_cutoff):
                    train_complete = True

                if (len(test_loss_all) > early_stopping_N):
                    ii = early_stopping_N + 1
                    if (min(test_loss_all[-early_stopping_N:]) >= test_loss_all[-ii]):
                        train_complete = True

            training_loss = 0
            if(not train_complete):
                self.train()
                for x, mask, y, _ in self.trainloader:
                    self.optimizer.zero_grad()
                    x, mask = x.clone().detach().to(self.device), mask.detach().to(self.device) 
                    y, z = y.clone().detach().to(self.device), self.network(x, mask)
                    loss = objective(y, z)
                    loss.backward()
                    self.optimizer.step()
                    training_loss += float(loss)

                training_loss = training_loss / (self.train_n)
                train_loss_all.append(training_loss)
                print("Training loss is " + format(training_loss))

            if (epoch % report_frequency == 0 and epoch > 0 and not train_complete):
                plot_index = [j * report_frequency for j in range(len(test_loss_all))]
                plt.plot(train_loss_all, label = 'Training loss')
                plt.plot(plot_index, test_loss_all, label = 'Testing loss')
                plt.plot(plot_index, full_loss_all, label = 'Full loss')
                plt.legend()
                plot_title = "All losses epochs " + format(epoch)
                plt.title(plot_title)
                path = "./loss_summaries/" + plot_title + ".png"
                plt.savefig(path)
                plt.clf()

        data_corrected_output = []
        self.eval()
        for x, mask, y, _ in self.loader:
            x, mask, y = x.clone().detach().to(self.device), mask.detach().to(self.device), y.clone().detach().to(self.device)
            z = (y - self.network(x, mask)).detach().cpu()

            data_corrected_output.append(z)

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