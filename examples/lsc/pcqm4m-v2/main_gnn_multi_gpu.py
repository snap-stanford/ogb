import torch
from torch_geometric.loader import DataLoader
from torch_geometric import seed_everything
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from gnn import GNN

import os
from tqdm.auto import tqdm
import argparse
import time

### importing OGB-LSC
try:
    from ogb.lsc import PygPCQM4Mv2Dataset, PCQM4Mv2Evaluator
except ImportError as error:
    print("`PygPCQM4Mv2Dataset` requires rdkit (`pip install rdkit`)")
    raise error

reg_criterion = torch.nn.L1Loss()

def train(model, rank, device, loader, optimizer):
    model.train()
    loss_accum = 0

    timer_frequency = 50

    iter_bar = tqdm(loader, desc="Warmup", disable=(rank > 0))

    total_time = 0
    last_time = None
    warmup = 200
    for step, batch in enumerate(iter_bar):
        batch = batch.to(device)

        pred = model(batch).view(-1,)
        optimizer.zero_grad()
        loss = reg_criterion(pred, batch.y)
        loss.backward()
        optimizer.step()

        loss_accum += loss.detach().cpu().item()

        if step > warmup and step % timer_frequency == 0:
            now = time.time()
            if last_time is not None:
                total_time += now - last_time
                iter_bar.set_description(f"Avg time per sample: {round(total_time/((step + 1 - warmup - timer_frequency)) * 1000, 2)} ms")
            last_time = now


    return loss_accum / (step + 1)

def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch).view(-1,)

        y_true.append(batch.y.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0)
    y_pred = torch.cat(y_pred, dim = 0)

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)["mae"]

def test(model, device, loader):
    model.eval()
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch).view(-1,)

        y_pred.append(pred.detach().cpu())

    y_pred = torch.cat(y_pred, dim = 0)

    return y_pred

def run(rank, dataset, args):
    num_devices = args.num_devices
    device = torch.device("cuda:" + str(rank)) if num_devices > 0 else torch.device("cpu")

    if num_devices > 1:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group("nccl", rank=rank, world_size=num_devices)

    split_idx = dataset.get_idx_split()

    train_idx = split_idx["train"]

    if num_devices > 1:
        train_idx = train_idx.split(train_idx.size(0) // num_devices)[rank]

    if args.train_subset:
        subset_ratio = 0.1
        subset_idx = torch.randperm(len(train_idx))[:int(subset_ratio*len(split_idx["train"]))]
        train_loader = DataLoader(dataset[train_idx[subset_idx]], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    else:
        train_loader = DataLoader(dataset[train_idx], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)

    if rank == 0:
        valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

        if args.save_test_dir != '':
            testdev_loader = DataLoader(dataset[split_idx["test-dev"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
            testchallenge_loader = DataLoader(dataset[split_idx["test-challenge"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

        if args.checkpoint_dir != '':
            os.makedirs(args.checkpoint_dir, exist_ok = True)
        ### automatic evaluator. takes dataset name as input
        evaluator = PCQM4Mv2Evaluator()

    shared_params = {
        'num_layers': args.num_layers,
        'emb_dim': args.emb_dim,
        'drop_ratio': args.drop_ratio,
        'graph_pooling': args.graph_pooling
    }

    if args.gnn == 'gin':
        model = GNN(gnn_type = 'gin', virtual_node = False, **shared_params)
    elif args.gnn == 'gin-virtual':
        model = GNN(gnn_type = 'gin', virtual_node = True, **shared_params)
    elif args.gnn == 'gcn':
        model = GNN(gnn_type = 'gcn', virtual_node = False, **shared_params)
    elif args.gnn == 'gcn-virtual':
        model = GNN(gnn_type = 'gcn', virtual_node = True, **shared_params)
    else:
        raise ValueError('Invalid GNN type')
    if num_devices > 0:
        model.to(rank)

    if num_devices > 1:
        model = DistributedDataParallel(model, device_ids=[rank])
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_params = sum(p.numel() for p in model.parameters())

    if rank == 0:
        print(f'#Params: {num_params}')

    if args.log_dir != '':
        writer = SummaryWriter(log_dir=args.log_dir)

    best_valid_mae = 1000

    if args.train_subset:
        scheduler = StepLR(optimizer, step_size=300, gamma=0.25)
        args.epochs = 1000
    else:
        scheduler = StepLR(optimizer, step_size=30, gamma=0.25)

    current_epoch = 1

    checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoint.pt')
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)

        current_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        best_valid_mae = checkpoint['best_val_mae']

        print(f"Found checkpoint, resume training at epoch {current_epoch}")

    for epoch in range(current_epoch, args.epochs + 1):
        if rank == 0:
            print("=====Epoch {}".format(epoch))
            print('Training...')
            start_time = time.time()
        train_mae = train(model, rank, device, train_loader, optimizer)
        if num_devices > 1:
            dist.barrier()

        if rank == 0:
            print(f"Training time for epoch {epoch}: {time.time()-start_time}s")
            print('Evaluating...')
            valid_mae = eval(model.module if isinstance(model, DistributedDataParallel) else model,
                             device,
                             valid_loader,
                             evaluator)

            print({'Train': train_mae, 'Validation': valid_mae})

            if args.log_dir != '':
                writer.add_scalar('valid/mae', valid_mae, epoch)
                writer.add_scalar('train/mae', train_mae, epoch)

            if valid_mae < best_valid_mae:
                best_valid_mae = valid_mae
                if args.checkpoint_dir != '':
                    print('Saving checkpoint...')
                    checkpoint = {'epoch': epoch,
                                  'model_state_dict': model.state_dict(),
                                  'optimizer_state_dict': optimizer.state_dict(),
                                  'scheduler_state_dict': scheduler.state_dict(),
                                  'best_val_mae': best_valid_mae,
                                  'num_params': num_params}
                    torch.save(checkpoint, checkpoint_path)

                if args.save_test_dir != '':
                    testdev_pred = test(model.module if isinstance(model, DistributedDataParallel) else model,
                                        device,
                                        testdev_loader)
                    testdev_pred = testdev_pred.cpu().detach().numpy()

                    testchallenge_pred = test(model, device, testchallenge_loader)
                    testchallenge_pred = testchallenge_pred.cpu().detach().numpy()

                    print('Saving test submission file...')
                    evaluator.save_test_submission({'y_pred': testdev_pred}, args.save_test_dir, mode = 'test-dev')
                    evaluator.save_test_submission({'y_pred': testchallenge_pred}, args.save_test_dir, mode = 'test-challenge')

            print(f'Best validation MAE so far: {best_valid_mae}')
        if num_devices > 1:
            dist.barrier()

        scheduler.step()


    if rank == 0 and args.log_dir != '':
        writer.close()


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on pcqm4m with Pytorch Geometrics')
    parser.add_argument('--gnn', type=str, default='gin-virtual',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--graph_pooling', type=str, default='sum',
                        help='graph pooling strategy mean or sum (default: sum)')
    parser.add_argument('--drop_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=600,
                        help='dimensionality of hidden units in GNNs (default: 600)')
    parser.add_argument('--train_subset', action='store_true')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--log_dir', type=str, default="",
                        help='tensorboard log directory')
    parser.add_argument('--checkpoint_dir', type=str, default = '', help='directory to save checkpoint')
    parser.add_argument('--save_test_dir', type=str, default = '', help='directory to save test submission file')
    parser.add_argument('--num_devices', type=int, default='0', help="Number of GPUs, if 0 runs on the CPU")
    args = parser.parse_args()

    print(args)

    seed_everything(42)

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    assert args.num_devices <= available_gpus, f"Cannot train with {args.num_devices} GPUs: available GPUs count {available_gpus}"

    ### automatic dataloading and splitting
    dataset = PygPCQM4Mv2Dataset(root = 'dataset/')

    if args.num_devices > 1:
        print(f'Starting multi-GPU training with DDP with {args.num_devices} GPUs')
        mp.spawn(run, args=(dataset, args), nprocs=args.num_devices, join=True)
    else:
        run(0, dataset, args)
