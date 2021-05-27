import torch


class EloNetwork(torch.nn.Module):
    ELO_CONSTANT = 400

    def __init__(self, num_models, initial_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.elo_vals = torch.nn.Linear(num_models - 1, 1)
        with torch.no_grad():
            self.elo_vals.bias.fill_(0.0)
            if initial_weights:
                self.elo_vals.weight.data = torch.FloatTensor(initial_weights)
        self.elo_vals.requires_grad = True

    def forward(self, batch):
        batch = batch.float()
        batch1, batch2 = torch.split(batch, 1, 1)
        r1 = self.elo_vals.forward(batch1)
        r2 = self.elo_vals.forward(batch2)

        q1 = torch.pow(10, r1 / self.ELO_CONSTANT)
        q2 = torch.pow(10, r2 / self.ELO_CONSTANT)

        expected = q1 / (q1 + q2)
        return expected

    def loss(self, expected, result):
        result_tensor = torch.tensor(result, requires_grad=False, dtype=torch.float)
        loss = torch.nn.functional.binary_cross_entropy(expected.view(-1), result_tensor)
        return loss