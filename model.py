from torch import nn


class TestModel(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, layer_size: int):
        super().__init__()
        self.h1 = nn.Sequential(
            nn.Linear(input_channels, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(),
        )
        self.h2 = nn.Sequential(
            nn.Linear(layer_size, output_channels),
        )

    def forward(self, data):
        x = self.h1(data)
        x = self.h2(x)
        return x


class Network(nn.Module):
    def __init__(
            self,
            input_channels: int = 784,
            output_channels: int = 10,
            layer_size: int = 32,
            layer_number: int = 3
    ):
        super().__init__()
        self.feature_extractor = nn.Sequential()

        input_layer = nn.Linear(input_channels, layer_size)
        nn.init.xavier_uniform(input_layer.weight)

        self.feature_extractor.add_module(
            f'layer_0',
            nn.Sequential(
                input_layer,
                nn.Tanh(),
            )
        )

        for i in range(1, layer_number):
            layer = nn.Linear(layer_size, layer_size)
            nn.init.xavier_uniform(layer.weight)
            hl = nn.Sequential(
                layer,
                nn.Tanh(),
            )
            self.feature_extractor.add_module(f'layer_{i}', hl)

        self.classifier = nn.Linear(layer_size, output_channels)
        nn.init.xavier_uniform(self.classifier.weight)

    def forward(self, data):
        x = self.feature_extractor(data)
        out = self.classifier(x)
        return out
