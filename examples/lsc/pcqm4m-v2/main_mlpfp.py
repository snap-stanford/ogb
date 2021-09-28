import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR

import os
import os.path as osp
from tqdm import tqdm
import argparse
import time
import numpy as np
import random

from rdkit import Chem
from rdkit.Chem import AllChem

### importing OGB-LSC
from ogb.lsc import PCQM4Mv2Dataset, PCQM4Mv2Evaluator

reg_criterion = torch.nn.L1Loss()

def train(model, device, loader, optimizer):
    model.train()
    loss_accum = 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        x, y = batch
        x = x.to(device).to(torch.float32)
        y = y.to(device)

        pred = model(x).view(-1,)
        optimizer.zero_grad()
        loss = reg_criterion(pred, y)
        loss.backward()
        optimizer.step()

        loss_accum += loss.detach().cpu().item()

    return loss_accum / (step + 1)

def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        x, y = batch
        x = x.to(device).to(torch.float32)
        y = y.to(device)

        with torch.no_grad():
            pred = model(x).view(-1,)

        y_true.append(y.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0)
    y_pred = torch.cat(y_pred, dim = 0)

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)["mae"]

def test(model, device, loader):
    model.eval()
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        x, y = batch 
        x = x.to(device).to(torch.float32)

        with torch.no_grad():
            pred = model(x).view(-1,)

        y_pred.append(pred.detach().cpu())

    y_pred = torch.cat(y_pred, dim = 0)

    return y_pred

class MLP(torch.nn.Module):
    def __init__(self, num_mlp_layers = 5, emb_dim = 300, drop_ratio = 0):
        super(MLP, self).__init__()
        self.num_mlp_layers = num_mlp_layers
        self.emb_dim = emb_dim
        self.drop_ratio = drop_ratio 

        # mlp
        module_list = [
            torch.nn.Linear(2048, self.emb_dim),
            torch.nn.BatchNorm1d(self.emb_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(p = self.drop_ratio),
        ]

        for i in range(self.num_mlp_layers - 1):
            module_list += [torch.nn.Linear(self.emb_dim, self.emb_dim),
            torch.nn.BatchNorm1d(self.emb_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(p = self.drop_ratio)]
        
        # relu is applied in the last layer to ensure positivity
        module_list += [torch.nn.Linear(self.emb_dim, 1)]

        self.mlp = torch.nn.Sequential(
            *module_list
        )
    
    def forward(self, x):
        output = self.mlp(x)
        if self.training:
            return output 
        else:
            # At inference time, we clamp the value between 0 and 20
            return torch.clamp(output, min=0, max=20)


def main_mlp():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--num_mlp_layers', type=int, default=6,
                        help='number of mlp layers (default: 6)')
    parser.add_argument('--drop_ratio', type=float, default=0.2,
                        help='dropout ratio (default: 0.2)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--emb_dim', type=int, default=1600,
                        help='embedding dimensionality (default: 1600)')
    parser.add_argument('--train_subset', action='store_true')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--radius', type=int, default=2,
                        help='radius (default: 2)')
    parser.add_argument('--log_dir', type=str, default="",
                        help='tensorboard log directory')
    parser.add_argument('--checkpoint_dir', type=str, default = '', help='directory to save checkpoint')
    parser.add_argument('--save_test_dir', type=str, default = '', help='directory to save test submission file')
    args = parser.parse_args()

    print(args)

    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    random.seed(42)

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    dataset = PCQM4Mv2Dataset(root='dataset/', only_smiles=True)
    fp_processed_file = preprocess_fp(dataset, args.radius)

    data_dict = torch.load(fp_processed_file)
    X, Y = data_dict['X'], data_dict['Y']
        
    split_idx = dataset.get_idx_split()
    ### automatic evaluator. takes dataset name as input
    evaluator = PCQM4Mv2Evaluator()

    if args.train_subset:
        print('train subset')
        subset_ratio = 0.1
        subset_idx = torch.randperm(len(split_idx["train"]))[:int(subset_ratio*len(split_idx["train"]))]
        train_dataset = TensorDataset(X[split_idx['train'][subset_idx]], Y[split_idx['train'][subset_idx]])

    else:
        train_dataset = TensorDataset(X[split_idx['train']], Y[split_idx['train']])

    valid_dataset = TensorDataset(X[split_idx['valid']], Y[split_idx['valid']])
    testdev_dataset = TensorDataset(X[split_idx['test-dev']], Y[split_idx['test-dev']])
    testchallenge_dataset = TensorDataset(X[split_idx['test-challenge']], Y[split_idx['test-challenge']])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    if args.save_test_dir != '':
        testdev_loader = DataLoader(testdev_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
        testchallenge_loader = DataLoader(testchallenge_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    if args.checkpoint_dir != '':
        os.makedirs(args.checkpoint_dir, exist_ok = True)

    model = MLP(num_mlp_layers=args.num_mlp_layers, emb_dim=args.emb_dim, drop_ratio=args.drop_ratio).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f'#Params: {num_params}')

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if args.log_dir != '':
        writer = SummaryWriter(log_dir=args.log_dir)

    best_valid_mae = 1000

    if args.train_subset:
        scheduler = StepLR(optimizer, step_size=300, gamma=0.25)
        args.epochs = 1000
    else:
        scheduler = StepLR(optimizer, step_size=30, gamma=0.25)

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        train_mae = train(model, device, train_loader, optimizer)

        print('Evaluating...')
        valid_mae = eval(model, device, valid_loader, evaluator)

        print({'Train': train_mae, 'Validation': valid_mae})

        if args.log_dir != '':
            writer.add_scalar('valid/mae', valid_mae, epoch)
            writer.add_scalar('train/mae', train_mae, epoch)

        if valid_mae < best_valid_mae:
            best_valid_mae = valid_mae
            if args.checkpoint_dir != '':
                print('Saving checkpoint...')
                checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'best_val_mae': best_valid_mae, 'num_params': num_params}
                torch.save(checkpoint, osp.join(args.checkpoint_dir, 'checkpoint.pt'))

            if args.save_test_dir != '':
                testdev_pred = test(model, device, testdev_loader)
                testdev_pred = testdev_pred.cpu().detach().numpy()

                testchallenge_pred = test(model, device, testchallenge_loader)
                testchallenge_pred = testchallenge_pred.cpu().detach().numpy()

                print('Saving test submission file...')
                evaluator.save_test_submission({'y_pred': testdev_pred}, args.save_test_dir, mode = 'test-dev')
                evaluator.save_test_submission({'y_pred': testchallenge_pred}, args.save_test_dir, mode = 'test-challenge')


        scheduler.step()
            
        print(f'Best validation MAE so far: {best_valid_mae}')

    if args.log_dir != '':
        writer.close()


def preprocess_fp(dataset, radius):
    fp_processed_dir = osp.join(dataset.folder, 'fp_processed')
    fp_processed_file = osp.join(fp_processed_dir, f'data_radius{radius}.pt')
    print(fp_processed_file)
    if not osp.exists(fp_processed_file):
        ### automatic dataloading and splitting
        os.makedirs(fp_processed_dir, exist_ok=True)

        x_list = []
        y_list = []
        for i in tqdm(range(len(dataset))):
            smiles, y = dataset[i]
            mol = Chem.MolFromSmiles(smiles)
            x = torch.tensor(list(AllChem.GetMorganFingerprintAsBitVect(mol, radius)), dtype=torch.int8)
            y_list.append(y)
            x_list.append(x)
        
        X = torch.stack(x_list)
        Y = torch.tensor(y_list)
        print(X)
        print(Y)
        print(X.shape)
        print(Y.shape)
        data_dict = {'X': X, 'Y': Y}
        torch.save(data_dict, fp_processed_file)

    return fp_processed_file
    

if __name__ == "__main__":
    main_mlp()
