import torch
import torch.nn as nn


class EFNet(nn.Module):
    def __init__(self, cate_nuniqs,
                 nume_size,
                 dim_emb,
                 num_classes=1,
                 dnn_dim=[400, 400, 400],
                 mlp_dim=[100],
                 cross_enhance=2,
                 cross_cate=2
                 ):
        """
        数据集criteo共有13个数值型特征，26个类别型特征
        hid_dims: 最后的MLP部分隐藏层每层的节点数,几个维度就是几层
        num_classes: 最后的输出维度，因为最后是一个概率值数字所以用1
        cate_nuniqs: cate_size类别特征的唯一值个数列表，也就是每个类别特征的vocab_size所组成的列表
        nume_size: numeral数值特征的个数，此版本不考虑数值型输入，即没有数值特征的情况
        dnn_dim: 特征增强的网络结构
        mlp_dim: 离散型特征高阶特征交叉MLP每层的节点数
        dim_dnn: enhance feature dnn的embedding向量维度
        """
        super(EFNet, self).__init__()

        self.cate_size = len(cate_nuniqs)  # get类别特征26
        self.nume_size = nume_size  # get数值型特征13
        self.cross_dim = self.cate_size * dim_emb + self.nume_size
        self.cross_enhance = cross_enhance
        self.cross_cate = cross_cate
        # 类别特征的embedding表示, cate_size -> dim_fi
        self.emb_cate = nn.ModuleList([nn.Embedding(voc_size, dim_emb) for voc_size in cate_nuniqs])

        """特征增强的MLP1"""
        self.mlp_dims = [self.cross_dim] + mlp_dim
        for i in range(1, len(self.mlp_dims)):
            setattr(self, 'linear_1ef' + str(i), nn.Linear(self.mlp_dims[i - 1], self.mlp_dims[i]))
            setattr(self, 'batchNorm_1ef' + str(i), nn.BatchNorm1d(self.mlp_dims[i]))
            setattr(self, 'activation_1ef' + str(i), nn.ReLU())
            setattr(self, 'dropout_1ef' + str(i), nn.Dropout(0))
        self.mlp_cross1 = nn.Linear(self.mlp_dims[-1], self.cross_dim)
        """特征交叉的cross1"""
        self.cross_enhance = cross_enhance + 1
        for i in range(1, self.cross_enhance):
            setattr(self, 'cross1_' + str(i), nn.Linear(self.cross_dim, self.cross_dim))

        """特征增强的MLP2"""
        for i in range(1, len(self.mlp_dims)):
            setattr(self, 'linear_2ef' + str(i), nn.Linear(self.mlp_dims[i - 1], self.mlp_dims[i]))
            setattr(self, 'batchNorm_2ef' + str(i), nn.BatchNorm1d(self.mlp_dims[i]))
            setattr(self, 'activation_2ef' + str(i), nn.ReLU())
            setattr(self, 'dropout_2ef' + str(i), nn.Dropout(0))
        self.mlp_cross2 = nn.Linear(self.mlp_dims[-1], self.cross_dim)
        """特征交叉的cross2"""
        for i in range(1, self.cross_enhance):
            setattr(self, 'cross2_' + str(i), nn.Linear(self.cross_dim, self.cross_dim))

        """final cross"""
        self.cross = cross_cate + 1
        for i in range(1, self.cross):
            setattr(self, 'cross_' + str(i), nn.Linear(self.cross_dim, self.cross_dim))

        """final dnn"""
        self.dnn_dims = [self.cross_dim] + dnn_dim
        for i in range(1, len(self.dnn_dims)):
            setattr(self, 'linear_dnn' + str(i), nn.Linear(self.dnn_dims[i - 1], self.dnn_dims[i]))
            setattr(self, 'batchNorm_dnn' + str(i), nn.BatchNorm1d(self.dnn_dims[i]))
            setattr(self, 'activation_dnn' + str(i), nn.ReLU())
            setattr(self, 'dropout_dnn' + str(i), nn.Dropout(0))
        self.dnn_cross = nn.Linear(self.dnn_dims[-1], self.cross_dim)

        """out"""
        self.mlp_linear = nn.Linear(self.dnn_dims[-1] + self.cross_dim, num_classes)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_sparse, x_dense) :
        """
        x_sparse: sparse features    [bs, cate_size]
        x_dense:  dense features     [bs, dense_size]
        """
        # [bs, 1, dim_emb] * cate_size -> [bs, cate_size * dim_fi]
        emb_cate = [emb(x_sparse[:, i].unsqueeze(1)) for i, emb in enumerate(self.emb_cate)]
        emb_cate = torch.cat(emb_cate, dim = 1)  # [bs, cate_size, dim_emb]
        emb_cate = torch.flatten(emb_cate, 1)
        emb_nume = x_dense
        emb_cate = torch.cat((emb_cate, emb_nume), dim = 1)
        """enhancing"""
        # cross part
        cross1_x0 = emb_cate
        cross1 = cross1_x0
        for i in range(1, self.cross_enhance):
            cross1 = cross1_x0 * getattr(self, 'cross1_' + str(i))(cross1) + cross1

        cross2_x0 = emb_cate
        cross2 = cross2_x0
        for i in range(1, self.cross_enhance):
            cross2 = cross2_x0 * getattr(self, 'cross1_' + str(i))(cross2) + cross2

        # mlp部分
        mlp_out1 = torch.flatten(emb_cate, 1)
        for i in range(1, len(self.mlp_dims)):
            mlp_out1 = getattr(self, 'linear_1ef' + str(i))(mlp_out1)
            mlp_out1 = getattr(self, 'batchNorm_1ef' + str(i))(mlp_out1)
            mlp_out1 = getattr(self, 'activation_1ef' + str(i))(mlp_out1)
            mlp_out1 = getattr(self, 'dropout_1ef' + str(i))(mlp_out1)  # [bs,mlp_dim]
        mlp_out1 = self.mlp_cross1(mlp_out1)  # [bs, mlp_dim] * [mlp_dim, cross_dim]
        # mlp_out1 = mlp_out1 + emb_cate
        mlp_out1 = torch.mul(mlp_out1, cross1)
        # mlp_out1 = mlp_out1 + cross1

        mlp_out2 = torch.flatten(emb_cate, 1)
        for i in range(1, len(self.mlp_dims)):
            mlp_out2 = getattr(self, 'linear_2ef' + str(i))(mlp_out2)
            mlp_out2 = getattr(self, 'batchNorm_2ef' + str(i))(mlp_out2)
            mlp_out2 = getattr(self, 'activation_2ef' + str(i))(mlp_out2)
            mlp_out2 = getattr(self, 'dropout_2ef' + str(i))(mlp_out2)
        mlp_out2 = self.mlp_cross2(mlp_out2)  # [bs, mlp_dim] * [mlp_dim, cross_dim]
        # mlp_out2 = mlp_out2 + emb_cate
        mlp_out2 = torch.mul(mlp_out2, cross2)
        # mlp_out2 = mlp_out2 + cross2
        """sigmoid and add"""
        weight = self.sigmoid(mlp_out2)
        enhance_fea = torch.mul(weight, emb_cate) + torch.mul((1 - weight), mlp_out1)  # [bs, cross, 1]
        """DNN part"""
        dnn_out = torch.flatten(enhance_fea, 1)
        # [bs, cross_dim] -> [bs, dnn_dims]
        for i in range(1, len(self.dnn_dims)):
            dnn_out = getattr(self, 'linear_dnn' + str(i))(dnn_out)
            dnn_out = getattr(self, 'batchNorm_dnn' + str(i))(dnn_out)
            dnn_out = getattr(self, 'activation_dnn' + str(i))(dnn_out)
            dnn_out = getattr(self, 'dropout_dnn' + str(i))(dnn_out)
        """cross part"""
        cross_0 = enhance_fea  # [bs, cross_dim]
        cross_out = cross_0
        for i in range(1, self.cross_cate):
            cross_out = cross_0 * getattr(self, 'cross_' + str(i))(cross_out) + cross_out

        """concatenation"""
        out = torch.cat((cross_out, dnn_out), dim = 1)

        out = self.mlp_linear(out)
        out = self.sigmoid(out)

        return out
