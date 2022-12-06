import torch
import math

from torch import nn
import torch.nn.functional as F


## Networks needed for the corrections
class SelfAttention(nn.Module):
    def __init__(self, emb, heads):
        super().__init__()

        ## This is the dimension of the embedding being used to code the letters (factors) found in the data.
        self.emb = emb

        ## This is the number of "attention heads". Each attention head is generating 3 matrices. Keys, Queries and Values.
        ## Queries and Keys are multiplied and passed to softmax, which generates a vector of positive "weights". 
        ## The weights are used to transform the input from x_1,...,x_n into y_1,...,y_n. The interpretation is that this 
        ## matrix can learn patterns within the sequential data. y_1 for instance can be interpreted as containing information on
        ## the interaction between x_1 and x_1,..., x_n. 

        ## Multiple "attention heads" then are transforming the data according to different weight matrices, so these different attention heads
        ## can in theory look for different interactions within the sequential data.
        self.heads = heads

        ## Each attention head has its own K, Q, V matrices. 
        self.tokeys = nn.Linear(emb, emb * heads, bias=False)
        self.toqueries = nn.Linear(emb, emb * heads, bias=False)
        self.tovalues = nn.Linear(emb, emb * heads, bias=False)

        ## Output from attention heads has the same dimensions as the input. The interpretation here is that this linear
        ## layer is combining all "patterns" extracted from each attention head.
        self.unifyheads = nn.Linear(heads * emb, emb)

    def forward(self, x, mask):

        ## b is the minibatch number. This is how many sequences are fed into the network
        ## t is the length of the sequences that are passed to the network. In our case, something like max peptide length in dataset.
        ## e will be the embedding dimension of the letters in the alphabet. Likely something like 5 or so, as there's only ~20 amino acids. (??)
        # b, t, e = x.size()
        x_dim = x.dim()
        e = x.size(x.dim()-1)
        t = x.size(x.dim()-2)
        b = x.size(x.dim()-3)

        ## Completely independent parameter from the input in principle.
        h = self.heads
        assert e == self.emb, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb})'

        ## The output from all attention heads is concatenated. So the sizes are reshaped to split into the
        ## number of heads h.
        view_args = []
        for index in range(0, x_dim-3):
            view_args = view_args + [x.size(index)]
        view_args = view_args + [b, t, h, e]
        keys    = self.tokeys(x)   .view(*view_args)
        queries = self.toqueries(x).view(*view_args)
        values  = self.tovalues(x) .view(*view_args)

        # compute scaled dot-product self-attention

        # - fold heads into the batch dimension
        ## The weight matrices are computed all together. This is why the keys, queries and values are concatenated.
        view_args = []
        for index in range(0, x_dim-3):
            view_args = view_args + [x.size(index)]
        view_args = view_args + [b * h, t, e]
        keys = keys.transpose(x_dim-2, x_dim-1).contiguous().view(*view_args)
        queries = queries.transpose(x_dim-2, x_dim-1).contiguous().view(*view_args)
        values = values.transpose(x_dim-2, x_dim-1).contiguous().view(*view_args)

        # - get dot product of queries and keys, and scale
        ## The matrix 'dot' represents the weights used when transforming the original input. 
        ## All the heads are contained here, in the first (zero-th) dimension of the tensor.
        dot = torch.matmul(queries, keys.transpose(x_dim-2, x_dim-1))
        dot = dot / math.sqrt(e) # dot contains b*h  t-by-t matrices with raw self-attention logits
        
        assert dot.size()[-3:] == (b*h, t, t), f'Matrix has size {dot.size()}, expected {(b*h, t, t)}.'

        mask = mask.repeat_interleave(h, x_dim-3)
        dot = F.softmax(dot - mask, dim = x_dim-1) # dot now has row-wise self-attention probabilities

        ## This line from the original code was causing an error. Seems to be an NA check. Will add later.
        ## OLD LINE - assert not former.util.contains_nan(dot[:, 1:, :]) # only the forst row may contain nan

        # apply the self attention to the values
        view_args = []
        for index in range(0, x_dim-3):
            view_args = view_args + [x.size(index)]
        view_args = view_args + [b, h, t, e]
        out = torch.matmul(dot, values).view(*view_args)

        # swap h, t back, unify heads
        ## The weight matrices are used to transform the original sequence of inputs. Here, we use the weight matrices
        ## from each attention head to transform the input vectors x_1,...,x_t into h * t many vectors y, each of dimension e. This is
        ## then expressed as b observations of t vectors, each of dimension h*e.
        view_args = []
        for index in range(0, x_dim-3):
            view_args = view_args + [x.size(index)]
        view_args = view_args + [b, t, h * e]
        out = out.transpose(x_dim-2, x_dim-1).contiguous().view(*view_args)

        ## Finally these vectors are all passed to a single linear layer to be compressed down into t vectors.
        return self.unifyheads(out)


## Uses the self attention above with dropout and normalization layers.
class Transformer(nn.Module):
    def __init__(self, emb, heads, ff_mult = 5, p_dropout = 0.1):

        super().__init__()

        self.emb = emb
        self.heads = heads
        self.ff_mult = ff_mult

        self.ff = nn.Sequential(
            nn.Linear(emb, ff_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_mult * emb, emb))

        self.attention = SelfAttention(emb, heads)

        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)

        self.dropout1 = nn.Dropout(p = p_dropout)
        self.dropout2 = nn.Dropout(p = p_dropout)

    def forward(self, x, mask):

        attended = self.attention(self.norm1(x), mask)
        attended = x + self.dropout1(attended)

        ## These are called reisudal connections. They are used in the transformer I'm working off of, as they seem to help performance.
        attended = self.ff(self.norm2(attended))
        x = x + self.dropout2(attended)

        return x, mask


## Used to chain transformers together. 
class mySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


## Transformer chain which takes an input x, mask_x.
class Transformer_Chain(nn.Module):
    def __init__(self, emb, depth, heads, ff_mult = 5):

        super().__init__()

        tblocks = []
        for i in range(depth):
            tblocks.append(Transformer(emb, heads, ff_mult))
        self.tblocks = mySequential(*tblocks)
      
    def forward(self, x, mask):
        x, mask = self.tblocks(x, mask)

        return(x)

        


## A block of transformers (determined by depth) followed by a correction layer. This correction
## layer is meant to output the batch corrections.
class TransformNet(nn.Module):
    def __init__(self, emb, seq_length, depth, n_batches, batch_size, heads = 5, ff_mult = 5):

        super().__init__()

        ## Networks
        self.transformers = Transformer_Chain(emb, depth, heads, ff_mult)
        self.correction = nn.Sequential(nn.Linear(emb * seq_length, ff_mult * emb), nn.ReLU(), 
                                        nn.Linear(ff_mult * emb, n_batches))
        
        self.batch_size = batch_size
           
    def forward(self, x, mask):

        x = self.transformers(x, mask)
        x_dim = x.dim()
        ## We take all the output from the transformer when making correction
        x = torch.flatten(x, x_dim-2, x_dim-1)

        x = self.correction(x)
        x = x.repeat_interleave(self.batch_size, x_dim-2)

        return x








######################################
## Testing masking in transformer chain
# testing = Transformer_Chain(3, 3)
#
# mask1 = torch.tensor([mask_helper(3, 3)])
# mask2 = torch.tensor([mask_helper(3, 6)])
#
# input = torch.rand((1, 6, 3))
#
# testing(input[:,0:3,:], mask1)
# testing(input, mask2)





