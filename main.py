# MINIPROJECT IN APPLIED STATISTICS
import pandas as pd
import torch
import random
import numpy as np
from config import CONFIG
from trainer import Trainer
from model import Network
from dataloaders import ProjectDataset
from datetime import datetime


torch.manual_seed(1)
random.seed(1)
np.random.seed(1)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True


def main():
    # Download dataset: https://www.kaggle.com/datasets/zalando-research/fashionmnist#:~:text=Fashion%2DMNIST%20is%20a%20dataset,a%20label%20from%2010%20classes.
    c = CONFIG()
    epochs = c.EPOCHS
    input_size = c.INPUT_SIZE
    output_classes = c.OUTPUT_CLASSES

    experiment_setup = pd.read_csv("setup/randomized_experiment.csv", delimiter=',')
    curated_setup = experiment_setup.drop(experiment_setup.columns[[0, 1]], axis=1)

    for row in curated_setup.iterrows():
        print(row[0])
        lr, bs, ls, ln = c.select_parameters(row[1])
        learning_rate = lr
        batch_size = bs
        layer_size = ls
        layer_number = ln

        print(f"Configuration completed.")
        print(f"Experimental setup: ")
        print(learning_rate)
        print(batch_size)
        print(layer_size)
        print(layer_number)

        training_data = ProjectDataset('data/fashion-mnist_train.csv', one_hot_encode=True)
        train_loader = torch.utils.data.DataLoader(
            training_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            drop_last=True
        )

        test_data = ProjectDataset('data/fashion-mnist_test.csv', one_hot_encode=True)
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )

        print(f"Data init completed.")
        model = Network(input_size, output_classes, layer_size, layer_number)
        print(f"Model init completed.")
        print(f"{model}")

        optim = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,)
        criterion = torch.nn.CrossEntropyLoss()

        trainer = Trainer(
            model=model,
            optim=optim,
            criterion=criterion,
            epochs=epochs,
        )
        trainer.train(train_loader)
        trainer.plot_loss(f"Run-{row[0]}_lr-{lr}_bs-{bs}_ls-{ls}_ln-{ln}")
        trainer.save_data(f"Run-{row[0]}_lr-{lr}_bs-{bs}_ls-{ls}_ln-{ln}")
        trainer.test(test_loader)
        experiment_setup.iloc[row[0], -1] = trainer.test_accuracy

    now = datetime.now()
    t = now.strftime("%Y-%m-%d")
    experiment_setup.to_csv(t + '_final_output.csv')


if __name__ == "__main__":
    main()
