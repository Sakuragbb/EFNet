import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from models import *
from utils import *
from sklearn.preprocessing import LabelEncoder

seed = 2022
torch.manual_seed(seed)


if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = 'cpu'
    print(device)
    # 1. Label encoding
    data = pd.read_csv('datasets/anime_train.csv')
    print(data.shape)
    sparse_fea = ['user_id', 'anime_id', 'name', 'genre', 'type', 'episodes', 'rating_y', 'members']
    dense_fea = []
    target = ['Label']
    data[sparse_fea] = data[sparse_fea].fillna('-1', )
    dim_emb = 10
    for fea in tqdm(sparse_fea):
        lbe = LabelEncoder()
        data[fea] = lbe.fit_transform(data[fea])

    # 2.count unique features for each sparse field and encode train and valid
    cate_uniques = [data[f].nunique() for f in sparse_fea]
    print(cate_uniques)
    train, valid = train_test_split(data, test_size = 0.1, random_state = 2022)
    print(train.shape, valid.shape)
    train_dataset = Data.TensorDataset(torch.LongTensor(train[sparse_fea].values),
                                       torch.FloatTensor(train[dense_fea].values),
                                       torch.FloatTensor(train['Label'].values), )

    valid_dataset = Data.TensorDataset(torch.LongTensor(valid[sparse_fea].values),
                                       torch.FloatTensor(valid[dense_fea].values),
                                       torch.FloatTensor(valid['Label'].values), )

    train_loader = Data.DataLoader(dataset = train_dataset, batch_size = 4096, shuffle = True)
    valid_loader = Data.DataLoader(dataset = valid_dataset, batch_size = 4096, shuffle = False)

    # 3. Define Model and Train
    # for cross in [1, 2, 3]:
    # for mlp in [[100], [200], [300], [400]]:
    #     for cross_enhance in [1, 2, 3]:
    # cross = 1
    mlp = [100]

    cross_enhance = 2
    #         model = DeFM_EF(cate_uniques, nume_size=len(dense_fea), dim_emb=dim_emb,
    #           cross_cate=cross, cross_enhance=cross_enhance, mlp_dim=mlp,dnn_dim =[400, 400, 400])
    # model = EFNet(cate_uniques, nume_size = len(dense_fea), dim_emb = dim_emb)
    model = EFNet(cate_uniques, nume_size = len(dense_fea), dim_emb = dim_emb)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    loss_fcn = nn.BCELoss()
    loss_fcn.to(device)
    # write_log("\ncross:{}, mlp:{}, enhance:{} " .format(cross, mlp, cross_enhance))
    # write_log("\nmlp:{}".format(mlp))
    train_and_eval(model, train_loader, valid_loader, 100, 100, device, optimizer, loss_fcn, scheduler)

