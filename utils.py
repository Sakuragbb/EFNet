import time, datetime
import torch
from sklearn.metrics import roc_auc_score


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def write_log(w):
    file_name = 'data/' + datetime.date.today().strftime('%m%d') + "_{}.log".format("criteo_hp")
    t0 = datetime.datetime.now().strftime('%H:%M:%S')
    info = "{} : {}".format(t0, w)
    print(info)
    with open(file_name, 'a') as f:
        f.write(info + '\n')

def heatm(w):
    file_name = 'data/' + "hot_value.csv"
    with open(file_name, 'a') as f:
        f.write(str(w))

def train_and_eval(model, train_loader, valid_loader, epochs, step, device, optimizer, loss_fcn, scheduler):
    best_auc = 0.0
    patience = 0
    best = 0.0
    for _ in range(epochs):
        """train"""
        model.train()
        print("Current lr : {}".format(optimizer.state_dict()['param_groups'][0]['lr']))
        write_log('Epoch: {}'.format(_ + 1))
        train_loss_sum = 0.0
        start_time = time.time()
        for idx, x in enumerate(train_loader):
            cate_fea, dense_fea, label = x[0], x[1], x[2]
            cate_fea, dense_fea, label = cate_fea.to(device), dense_fea.to(device), label.to(device)
            pred = model(cate_fea, dense_fea)
            pred = pred.view(-1)
            # print(pred)
            loss = loss_fcn(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.cpu().item()
            if (idx + 1) % step == 0 or (idx + 1) == len(train_loader):
                write_log("epoch {:04d} | step {:04d} / {} | loss {:.6f} | time {:.6f}"
                          .format(_ + 1, idx + 1, len(train_loader),
                                  train_loss_sum / (idx + 1), time.time() - start_time))
        scheduler.step()
        """eval"""

        with torch.no_grad():
            valid_labels, valid_preds = [], []
            for idx, x in enumerate(valid_loader):
                cate_fea, dense_fea, label = x[0], x[1], x[2]
                cate_fea, dense_fea = cate_fea.to(device), dense_fea.to(device)
                pred = model(cate_fea, dense_fea).reshape(-1).data.cpu().numpy().tolist()
                valid_preds.extend(pred)
                valid_labels.extend(label.cpu().numpy().tolist())
        current_auc = roc_auc_score(valid_labels, valid_preds)
        if current_auc >= best_auc:
            best_auc = current_auc
            patience = 0
            if best_auc >= best:
                best = best_auc
                torch.save(model.state_dict(), "data/EFNET_best.pth")
        else:
            patience = patience + 1
            if patience >= 3:
                break

        write_log('current AUC: %.6f, best AUC: %6f\n' % (current_auc, best_auc))

def htmp(model, train_loader, valid_loader, epochs, step, device, optimizer, loss_fcn, scheduler):
    best_auc = 0.0
    patience = 0
    best = 0.0
    for _ in range(epochs):
        """train"""
        model.train()
        print("Current lr : {}".format(optimizer.state_dict()['param_groups'][0]['lr']))
        write_log('Epoch: {}'.format(_ + 1))
        train_loss_sum = 0.0
        start_time = time.time()
        for idx, x in enumerate(train_loader):
            cate_fea, dense_fea, label = x[0], x[1], x[2]
            cate_fea, dense_fea, label = cate_fea.to(device), dense_fea.to(device), label.to(device)
            pred, fea, enfea = model(cate_fea, dense_fea)
            pred = pred.view(-1)
            # print(pred)
            loss = loss_fcn(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.cpu().item()
            if (idx + 1) % step == 0 or (idx + 1) == len(train_loader):
                write_log("epoch {:04d} | step {:04d} / {} | loss {:.6f} | time {:.6f}"
                          .format(_ + 1, idx + 1, len(train_loader),
                                  train_loss_sum / (idx + 1), time.time() - start_time))
        scheduler.step()
        """eval"""

        with torch.no_grad():
            valid_labels, valid_preds = [], []
            for idx, x in enumerate(valid_loader):
                cate_fea, dense_fea, label = x[0], x[1], x[2]
                cate_fea, dense_fea = cate_fea.to(device), dense_fea.to(device)
                pred, fea, enfea = model(cate_fea, dense_fea)
                pred = pred.reshape(-1).data.cpu().numpy().tolist()
                valid_preds.extend(pred)
                valid_labels.extend(label.cpu().numpy().tolist())
        current_auc = roc_auc_score(valid_labels, valid_preds)
        if current_auc >= best_auc:
            best_auc = current_auc
            patience = 0
            if best_auc >= best:
                best = best_auc
                if _ >= 10:
                    fea = torch.narrow(fea, 0, 0, 1)
                    enfea = torch.narrow(fea, 0, 0, 1)
                    heatm(fea.cpu().tolist() + enfea.cpu().tolist())

        else:
            patience = patience + 1
            if patience >= 3:
                break

        write_log('current AUC: %.6f, best AUC: %6f\n' % (current_auc, best_auc))
