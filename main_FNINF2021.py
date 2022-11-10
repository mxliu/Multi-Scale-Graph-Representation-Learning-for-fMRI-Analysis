


import os
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim 
import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
import h5py

datax1 =h5py.File('data\\116.mat')
datax2 =h5py.File('data\\200.mat')
class Dataset116(object):
    

    def __init__(self, data_root="data", train_size=0.8):
        self.data_root = data_root
        sparse_adjacency, node_labels, graph_indicator, graph_labels = self.read_data()
        self.sparse_adjacency = sparse_adjacency.tocsr()
        self.node_labels = node_labels
        self.graph_indicator = graph_indicator
        self.graph_labels = graph_labels

        self.train_index, self.test_index = self.split_data(train_size)
        self.train_label = graph_labels[self.train_index]  
        self.test_label = graph_labels[self.test_index]  

    def split_data(self, train_size):
        unique_indicator = np.asarray(list(set(self.graph_indicator)))
       
        # print(unique_indicator)
        train_index, test_index = train_test_split(unique_indicator,
                                                   train_size=train_size,
                                                   random_state=777)
        return train_index, test_index

    def __getitem__(self, index):
        mask = self.graph_indicator == index
        graph_indicator = self.graph_indicator[mask]
        node_labels = self.node_labels[mask]
        graph_labels = self.graph_labels[index]
        adjacency = self.sparse_adjacency[mask, :][:, mask]
        # print(adjacency.size())
        return adjacency, node_labels, graph_indicator, graph_labels

    def __len__(self):
        return len(self.graph_labels)

    def read_data(self):

        data_dir = os.path.join(self.data_root, "data_116")
        print("Loading 116_node_labels.txt")
       
        node_labels = np.genfromtxt(os.path.join(data_dir, "116_node_labels.txt"),
                                    dtype=np.int64) - 1
        print("Loading 116_graph_indicator.txt")
        
        graph_indicator = np.genfromtxt(os.path.join(data_dir, "116_graph_indicator.txt"),
                                        dtype=np.int64) - 1
        print("Loading 116_graph_labels.txt")

        graph_labels = np.genfromtxt(os.path.join(data_dir, "116_graph_labels.txt"),
                                     dtype=np.int64) - 1
        num_nodes = len(node_labels)
       
        sparse_adjacency = sp.coo_matrix(datax1['adi116'],
                                         shape=(num_nodes, num_nodes), dtype=np.float32)
        return sparse_adjacency, node_labels, graph_indicator, graph_labels


class Dataset200(object):
   

    def __init__(self, data_root="data"):
        self.data_root = data_root

        sparse_adjacency, node_labels, graph_indicator, graph_labels = self.read_data()

        self.sparse_adjacency = sparse_adjacency.tocsr()
        self.node_labels = node_labels
        self.graph_indicator = graph_indicator
        self.graph_labels = graph_labels


    def __getitem__(self, index):
        mask = self.graph_indicator == index

        graph_indicator = self.graph_indicator[mask]

        node_labels = self.node_labels[mask]

        graph_labels = self.graph_labels[index]

        adjacency = self.sparse_adjacency[mask, :][:, mask]
        return adjacency, node_labels, graph_indicator, graph_labels

    def __len__(self):
        return len(self.graph_labels200)

    def read_data(self):

        data_dir = os.path.join(self.data_root, "data_200")

        # print (adjacency_list)
        print("Loading 200_node_labels.txt")

        node_labels = np.genfromtxt(os.path.join(data_dir, "200_node_labels.txt"),
                                    dtype=np.int64) - 1
        print("Loading 200_graph_indicator.txt")

        graph_indicator = np.genfromtxt(os.path.join(data_dir, "200_graph_indicator.txt"),
                                        dtype=np.int64) - 1
        print("Loading 200_graph_labels.txt")

        graph_labels = np.genfromtxt(os.path.join(data_dir, "200_graph_labels.txt"),
                                     dtype=np.int64) - 1
        num_nodes = len(node_labels)
       
        sparse_adjacency = sp.coo_matrix(datax2['adi200'],
                                         shape=(num_nodes, num_nodes), dtype=np.float32)
        return sparse_adjacency, node_labels, graph_indicator, graph_labels


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):

        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim 
        self.output_dim = output_dim 
        self.use_bias = use_bias
        #
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters() #
 
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)
 
    def forward(self, adjacency, input_feature):
        
        support = torch.mm(input_feature, self.weight) #XW (N,output_dim=hidden_dim)
        output = torch.sparse.mm(adjacency, support) #L(XW)  (N,output_dim=hidden_dim)
        if self.use_bias:
            output += self.bias
        return output #(N,output_dim=hidden_dim)
 

            
            
            



def tensor_from_numpy(x, device): 
    return torch.from_numpy(x).to(device)
 
 
def normalization(adjacency):

    adjacency += sp.eye(adjacency.shape[0])    
    degree = np.array(adjacency.sum(1)) 
    d_hat = sp.diags(np.power(degree, -0.5).flatten()) 
    L = d_hat.dot(adjacency).dot(d_hat).tocoo() 
    
    indices = torch.from_numpy(np.asarray([L.row, L.col])).long()
   
    values = torch.from_numpy(L.data.astype(np.float32))
   
    tensor_adjacency = torch.sparse.FloatTensor(indices, values, L.shape)
    
    return tensor_adjacency
 
 

    


class Model116(nn.Module):
    def __init__(self, input_dim, hidden_dim):

        super(Model116, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gcn1 = GraphConvolution(input_dim, hidden_dim)
        self.gcn2 = GraphConvolution(hidden_dim, hidden_dim)
        
       
 
    def forward(self, adjacency, input_feature):
        #adjacency (N,N)
        #input_feature (N,input_dim)
        gcn1 = F.relu(self.gcn1(adjacency, input_feature)) #(N,hidden_dim)
        gcn2 = F.relu(self.gcn2(adjacency, gcn1))#(N,hidden_dim)
        
        gcn2= gcn2.reshape(-1,116,32)   
                           
        readout=torch.mean(gcn2,1)#
        readout2=torch.amax(gcn2,1)
       
        fc116= torch.cat((readout, readout2), dim=1)
        
        return fc116
class Model200(nn.Module):
    def __init__(self, input_dim, hidden_dim):
       
        super(Model200, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gcn1 = GraphConvolution(input_dim, hidden_dim)
        self.gcn2 = GraphConvolution(hidden_dim, hidden_dim)
       
        
 
    def forward(self, adjacency, input_feature):
        
        gcn1 = F.relu(self.gcn1(adjacency, input_feature)) #(N,hidden_dim)
        gcn2 = F.relu(self.gcn2(adjacency, gcn1))#(N,hidden_dim)
        gcn2= gcn2.reshape(-1,200,32)
                     
        readout=torch.mean(gcn2,1)#
        readout2=torch.amax(gcn2,1)
       
        fc200= torch.cat((readout, readout2), dim=1)
       
        
        return fc200  
    
    
      

class Model(nn.Module):
    def __init__(self, input_dim1,input_dim2,hidden_dim, num_classes=2):
        super(Model, self).__init__()
        self.input_dim1 = input_dim1
        self.hidden_dim1 = hidden_dim*2
        self.input_dim2 = input_dim2
        self.hidden_dim2 = hidden_dim*2
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.fc116=Model116(input_dim1, hidden_dim)
        self.fc200=Model200(input_dim2, hidden_dim)
        self.fc1 = nn.Sequential(nn.Linear(self.hidden_dim1+ self.hidden_dim2, hidden_dim),nn.Dropout(p=0.5))
        self.fc2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2),nn.Dropout(p=0.5))
        self.fc3 = nn.Sequential(nn.Linear(hidden_dim//2, num_classes))
    def forward(self,normalize_adjacency1, node_features1, normalize_adjacency2, node_features2):
        fc116  = self.fc116(normalize_adjacency1, node_features1)
        fc200  = self.fc200(normalize_adjacency2, node_features2)
        x =torch.cat((fc116,fc200),dim=1)
      
        fc1 = F.relu(self.fc1(x))
        fc2 = F.relu(self.fc2(fc1))
       
        logits = self.fc3(fc2)
        
        return logits
    
dataset116 = Dataset116()
dataset200 = Dataset200()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#The large adjacency matrix corresponding to all graphs


adjacency1 = dataset116.sparse_adjacency
#Normalization
normalize_adjacency1= normalization(adjacency1).to(DEVICE)
#Feature labels of all nodes
node_labels = tensor_from_numpy(dataset116.node_labels, DEVICE)
node_features1 = datax1['pc_sub']
node_features1=np.transpose(node_features1)
node_features1=torch.Tensor(node_features1).type(torch.float32).to(DEVICE)
#Category label for each graph
graph_labels1 = tensor_from_numpy(dataset116.graph_labels, DEVICE)
train_index = tensor_from_numpy(dataset116.train_index, DEVICE)
test_index = tensor_from_numpy(dataset116.test_index, DEVICE)
train_label = tensor_from_numpy(dataset116.train_label, DEVICE)
test_label = tensor_from_numpy(dataset116.test_label, DEVICE)


adjacency2 = dataset200.sparse_adjacency
normalize_adjacency2 = normalization(adjacency2).to(DEVICE)
node_labels = tensor_from_numpy(dataset200.node_labels, DEVICE)
node_features2 = datax2['pc_sub']
node_features2=np.transpose(node_features2)
node_features2=torch.Tensor(node_features2).to(DEVICE)
graph_labels2 = tensor_from_numpy(dataset200.graph_labels, DEVICE)




input_dim1 = node_features1.size(1)
hidden_dim1 = 32
input_dim2 = node_features2.size(1)
hidden_dim2 =    32# @param {type: "integer"}
hidden_dim =   32
num_classes = 2

# @param {type: "integer"}
LEARNING_RATE = 0.01 # @param
WEIGHT_DECAY = 0.00001 # @param


# Model initialization
model = Model(input_dim1,input_dim2, hidden_dim,num_classes).to(DEVICE)
EPOCHS=50
criterion = nn.CrossEntropyLoss().to(DEVICE)
#Adam
optimizer = optim.Adam(model.parameters(), LEARNING_RATE, weight_decay=WEIGHT_DECAY)
 
model.train() #
for epoch in range(EPOCHS):
    logits = model(normalize_adjacency1, node_features1,normalize_adjacency2,node_features2).to(DEVICE) #
    loss = criterion(logits[train_index], train_label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()  #
    #
    train_acc = torch.eq(
        logits[train_index].max(1)[1], train_label).float().mean()
    print("Epoch {:03d}: Loss {:.4f}, TrainAcc {:.4}".format(
        epoch, loss.item(), train_acc.item()))

def test():

    model.eval()  
    with torch.no_grad():  
        logits = model(normalize_adjacency1, node_features1, normalize_adjacency2,node_features2).to(DEVICE) #对所有数据(图)前向传播 得到输出
        test_logits = logits[test_index]  
        test_acc = torch.eq(test_logits.max(1)[1], test_label).float().mean()
       

    return test_acc


test_acc= test()
print(test_acc.item())

