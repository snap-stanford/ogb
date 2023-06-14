import argparse
import glob
import os
import os.path as osp
import pathlib
import time
from typing import Dict, List, NamedTuple, Optional, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch_geometric.transforms as T
from root import ROOT
from torch import Tensor
from torch.nn import BatchNorm1d, Dropout, Linear, ModuleList, ReLU, Sequential
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import StepLR
from torch_geometric import seed_everything
from torch_geometric.data import Batch, NeighborSampler
from torch_geometric.data.lightning import LightningNodeData
from torch_geometric.loader.neighbor_loader import NeighborLoader
from torch_geometric.nn import GATConv, SAGEConv, to_hetero
from torch_geometric.typing import Adj, EdgeType, NodeType
from torchmetrics import Accuracy
from tqdm import tqdm

from ogb.lsc import MAG240MDataset


class MAG240M(LightningNodeData):
    def __init__(self, *args, **kwargs):
        super(MAG240M, self).__init__(*args, **kwargs)

    @property
    def num_features(self) -> int:
        return 768

    @property
    def num_classes(self) -> int:
        return 153

    def metadata(self) -> Tuple[List[NodeType], List[EdgeType]]:
        node_types = ["paper", "author", "institution"]
        edge_types = [
            ("author", "affiliated_with", "institution"),
            ("institution", "rev_affiliated_with", "author"),
            ("author", "writes", "paper"),
            ("paper", "rev_writes", "author"),
            ("paper", "cites", "paper"),
        ]
        return node_types, edge_types


class HomoGNN(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        num_layers: int,
        heads: int = 4,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers

        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        x = x.to(torch.float)
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin(x)


class HeteroGNN(torch.nn.Module):
    def __init__(
        self,
        metadata: Tuple[List[NodeType], List[EdgeType]],
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        num_nodes_dict: Dict[NodeType, int],
        num_layers: int,
        heads: int = 4,
        dropout: float = 0.5,
        debug: bool = False,
    ):
        super().__init__()
        model = HomoGNN(
            in_channels,
            out_channels,
            hidden_channels,
            num_layers,
            heads=heads,
            dropout=dropout,
        )
        self.model = to_hetero(model, metadata, aggr="sum", debug=debug)
        # self.embeds = {}
        # for node_type, num_nodes in num_nodes_dict.items():
        #     if node_type != 'paper':
        #         self.embeds[node_type] = torch.nn.Embedding(num_embeddings=num_nodes, embedding_dim=in_channels)
        self.train_acc = Accuracy(task="multiclass", num_classes=out_channels)
        self.val_acc = Accuracy(task="multiclass", num_classes=out_channels)

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Tensor],
    ) -> Dict[NodeType, Tensor]:
        return self.model(x_dict, edge_index_dict)

    def common_step(self, batch: Batch) -> Tuple[Tensor, Tensor]:
        batch_size = batch["paper"].batch_size
        # for node_type, embed in self.embeds.items():
        #     batch[node_type].x = embed(batch[node_type].n_id)
        # w/o this to_hetero model fails
        for node_type in batch.node_types:
            if node_type not in batch.x_dict.keys():
                paper_x = batch["paper"].x
                # (TODO) replace this w/ embeddings for better learning once its working
                # (NOTE) embeddings take too much memory
                batch[node_type].x = torch.zeros(
                    size=(torch.numel(batch[node_type].n_id), paper_x.size(-1)),
                    device=paper_x.device,
                    dtype=paper_x.dtype,
                )
        y_hat = self(batch.x_dict, batch.edge_index_dict)["paper"][:batch_size]
        y = batch["paper"].y[:batch_size].to(torch.long)
        return y_hat, y

    def training_step(self, batch: Batch) -> Tensor:
        y_hat, y = self.common_step(batch)
        train_loss = F.cross_entropy(y_hat, y)
        self.train_acc(y_hat.softmax(dim=-1), y)
        return train_loss

    def validation_step(self, batch: Batch):
        y_hat, y = self.common_step(batch)
        return self.val_acc(y_hat.softmax(dim=-1), y)

    def predict_step(self, batch: Batch):
        y_hat, y = self.common_step(batch)
        return y_hat


def run(
    rank,
    data,
    n_devices=1,
    num_epochs=1,
    num_steps_per_epoch=100,
    log_every_n_steps=1,
    batch_size=1024,
    sizes=[128],
    hidden_channels=1024,
    dropout=0.5,
    eval_steps=100,
    num_warmup_iters_for_timing=10,
    debug=False,
):
    if rank == 0:
        print("Setting up...")
    since_setup = time.time()
    seed_everything(12345)
    if n_devices > 1:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group("nccl", rank=rank, world_size=n_devices)
    model = HeteroGNN(
        data.metadata(),
        data["paper"].x.size(-1),
        data.num_classes,
        hidden_channels,
        num_nodes_dict=data.collect("num_nodes"),
        num_layers=len(sizes),
        dropout=dropout,
        debug=debug,
    )
    if n_devices > 0:
        model.to(rank)
    if rank == 0:
        print(f"# GNN Params: {sum([p.numel() for p in model.parameters()])/10**6}M")

    train_idx = data["paper"].train_mask.nonzero(as_tuple=False).view(-1)
    if n_devices > 1:
        # Split training indices into `n_devices` many chunks:
        train_idx = train_idx.split(train_idx.size(0) // n_devices)[rank]
    eval_idx = data["paper"].val_mask.nonzero(as_tuple=False).view(-1)
    test_idx = data["paper"].test_mask.nonzero(as_tuple=False).view(-1)

    kwargs = dict(
        batch_size=batch_size, num_workers=get_num_workers(max(n_devices, 1)), persistent_workers=True
    )
    train_loader = NeighborLoader(
        data,
        input_nodes=("paper", train_idx),
        num_neighbors=sizes,
        shuffle=True,
        drop_last=True,
        **kwargs,
    )

    if rank == 0:
        eval_loader = NeighborLoader(
            data,
            input_nodes=("paper", eval_idx),
            num_neighbors=sizes,
            shuffle=True,
            **kwargs,
        )
        test_loader = NeighborLoader(
            data,
            input_nodes=("paper", test_idx),
            num_neighbors=sizes,
            shuffle=True,
            **kwargs,
        )
    if n_devices > 1:
        model = DistributedDataParallel(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, num_epochs + 1):
        model.train()
        time_sum = 0
        for i, batch in enumerate(train_loader):
            if i >= num_steps_per_epoch:
                break
            if rank == 0 and epoch == 0 and i == 0:
                print(f"Time to finish setup: {time.time() - since_setup:.4f}")
                print("Training beginning...")
            since = time.time()
            optimizer.zero_grad()
            if n_devices > 0:
                batch = batch.to(rank, "x", "y", "edge_index")
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            iter_time = time.time() - since
            if i > num_warmup_iters_for_timing:
                time_sum += iter_time
            if rank == 0 and i % log_every_n_steps == 0:
                print(
                    f"Epoch: {epoch:02d}, Step: {i:d}, Loss: {loss:.4f}, Step Time: {iter_time:.4f}s"
                )
        if n_devices > 1:
            dist.barrier()
        if rank == 0:
            print(
                f"Epoch: {epoch:02d}, Loss: {loss:.4f}, Average Step Time: {time_sum/(num_steps_per_epoch - num_warmup_iters_for_timing):.4f}s"
            )
            model.eval()
            acc_sum = 0
            with torch.no_grad():
                for i, batch in enumerate(eval_loader):
                    if i >= eval_steps:
                        break
                    if n_devices > 0:
                        batch = batch.to(rank, "x", "y", "edge_index")
                    acc_sum += model.validation_step(batch)
                print(
                    f"Validation Accuracy: {acc_sum/eval_steps * 100.0:.4f}%",
                )
    if n_devices > 1:
        dist.barrier()
    if rank == 0:
        model.eval()
        acc_sum = 0
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if i >= eval_steps:
                    break
                if n_devices > 0:
                    batch = batch.to(rank, "x", "y", "edge_index")
                acc_sum += model.validation_step(batch)
            print(
                f"Test Accuracy: {acc_sum/eval_steps * 100.0:.4f}%",
            )
    if n_devices > 1:
        dist.destroy_process_group()


def get_num_workers(world_size):
    num_work = None
    if hasattr(os, "sched_getaffinity"):
        try:
            num_work = len(os.sched_getaffinity(0)) / (2 * world_size)
        except Exception:
            pass
    if num_work is None:
        num_work = os.cpu_count() / (2 * world_size)
    return int(num_work)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_channels", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--num_steps_per_epoch", type=int, default=100)
    parser.add_argument("--log_every_n_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--num_warmup_iters_for_timing", type=int, default=10)
    parser.add_argument("--sizes", type=str, default="128")
    parser.add_argument(
        "--n_devices", type=int, default=1, help="0 devices for CPU, or 1-8 to use GPUs"
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    args.sizes = [int(i) for i in args.sizes.split("-")]
    print(args)
    if not args.debug:
        import warnings

        warnings.simplefilter("ignore")
    if not torch.cuda.is_available():
        print("No GPUs available, running with CPU")
        args.n_devices = 0
    if args.n_devices > torch.cuda.device_count():
        print(
            args.n_devices,
            "GPUs requested but only",
            torch.cuda.device_count(),
            "GPUs available",
        )
        args.n_devices = torch.cuda.device_count()
    dataset = MAG240MDataset(ROOT)
    data = dataset.to_pyg_hetero_data()
    if args.n_devices > 1:
        print("Let's use", args.n_devices, "GPUs!")
        mp.spawn(
            run,
            args=(
                data,
                args.n_devices,
                args.epochs,
                args.num_steps_per_epoch,
                args.log_every_n_steps,
                args.batch_size,
                args.sizes,
                args.hidden_channels,
                args.dropout,
                args.eval_steps,
                args.num_warmup_iters_for_timing,
                args.debug,
            ),
            nprocs=args.n_devices,
            join=True,
        )
    else:
        if args.n_devices == 1:
            print("Using a single GPU")
        else:
            print("Using CPU")
        run(
            0,
            data,
            args.n_devices,
            args.epochs,
            args.num_steps_per_epoch,
            args.log_every_n_steps,
            args.batch_size,
            args.sizes,
            args.hidden_channels,
            args.dropout,
            args.eval_steps,
            args.num_warmup_iters_for_timing,
            args.debug,
        )
