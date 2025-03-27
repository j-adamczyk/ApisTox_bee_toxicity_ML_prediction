from typing import Final, Optional

from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch import Tensor, nn
from torch_geometric.nn import (
    GAT,
    MLP,
    GINEConv,
    MessagePassing,
    global_add_pool,
    AttentiveFP,
)
from torch_geometric.nn.models.basic_gnn import GCN, BasicGNN, GraphSAGE
from torch_geometric.typing import OptTensor

from src.gnns.utils import set_determinism


class ClassicGNN(nn.Module):
    def __init__(
        self,
        conv_type: str = "GCN",
        num_layers: int = 3,
        num_channels: int = 64,
        dropout: float = 0.0,
        num_classes: int = 2,
        random_state: int = 0,
        *args,
        **kwargs,
    ):
        super().__init__()
        set_determinism(random_state)

        self.conv_type = conv_type
        if conv_type == "AttentiveFP":
            self.gnn = AttentiveFP(
                in_channels=num_channels,
                hidden_channels=num_channels,
                out_channels=num_channels,
                edge_dim=num_channels,
                num_layers=num_layers,
                num_timesteps=2,
                dropout=dropout,
            )
        else:
            # due to Jumping Knowledge concatenation
            out_channels = num_channels * num_layers

            if conv_type == "GCN":
                gnn_cls = GCN
            elif conv_type == "GraphSAGE":
                gnn_cls = GraphSAGE
            elif conv_type == "GIN":
                gnn_cls = GINE
            elif conv_type == "GAT":
                gnn_cls = GAT
            else:
                raise ValueError(f"conv_type {conv_type} not supported")

            self.gnn = gnn_cls(
                in_channels=num_channels,
                hidden_channels=num_channels,
                num_layers=num_layers,
                out_channels=out_channels,
                dropout=dropout,
                norm="LayerNorm",
                jk="cat",
            )

        self.atom_encoder = AtomEncoder(num_channels)
        self.bond_encoder = BondEncoder(num_channels)

        self.classifier = nn.Linear(out_channels, out_features=num_classes)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: OptTensor = None,
        edge_attr: OptTensor = None,
        batch: OptTensor = None,
        batch_size: Optional[int] = None,
        *args,
        **kwargs,
    ):
        x = self.atom_encoder(x)
        edge_attr = self.bond_encoder(edge_attr)
        if self.conv_type == "AttentiveFP":
            x = self.gnn(x, edge_index, edge_attr, batch)
        else:
            x = self.gnn(
                x,
                edge_index,
                edge_weight,
                edge_attr,
                batch,
                batch_size,
                *args,
                **kwargs,
            )
            x = global_add_pool(x, batch)

        y = self.classifier(x)
        return y


class GINE(BasicGNN):
    supports_edge_weight: Final[bool] = False
    supports_edge_attr: Final[bool] = True
    supports_norm_batch: Final[bool]

    def init_conv(
        self, in_channels: int, out_channels: int, **kwargs
    ) -> MessagePassing:
        mlp = MLP(
            [in_channels, out_channels, out_channels],
            act=self.act,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )
        return GINEConv(mlp, **kwargs)
