import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import json
import numpy as np
from torch import nn
from datetime import datetime
from sklearn.metrics import accuracy_score


class Trainer:
    def __init__(
            self,
            model: nn.Module,
            optim,
            criterion: nn.Module,
            epochs: int,
    ):
        self.model = model
        self.optim = optim
        self.criterion = criterion
        # self.optim = torch.optim.SGD(model.parameters(), lr=learning_rate)
        # self.criterion = torch.nn.BCEWithLogitsLoss()
        self.epochs = epochs

        self.global_step = 0

        self.train_loss = []
        self.train_accuracies = []
        self.test_accuracy = 0.0

        self.train_acc_steps = {}
        self.train_loss_steps = {}

        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def train(self, dataloader):
        print(f"Start training.")
        self.model.train()
        for epoch in range(self.epochs):
            accuracy_vals = []
            loss_vals = []
            for x_batch, y_batch in dataloader:
                loss, acc = self.train_step(x_batch, y_batch)
                accuracy_vals.append(acc)
                loss_vals.append(loss)
                self.global_step += 1

            self.train_acc_steps[self.global_step] = accuracy_vals
            self.train_loss_steps = loss_vals
            avg_acc = np.array(accuracy_vals).sum() / len(dataloader)
            avg_loss = np.array(loss_vals).sum() / len(dataloader)
            self.train_accuracies.append(avg_acc)
            self.train_loss.append(avg_loss)
            print(f"Epoch {epoch}: Average accuracy at {avg_acc}.")
            print(f"Epoch {epoch}: Average loss at {avg_loss}.")
        print(f"Training finished.")

    def test(self, dataloader):
        self.model.eval()
        self.test_accuracy = 0.0
        with torch.no_grad():
            for x, y in dataloader:
                self.test_accuracy += self.test_step(x, y)
            self.test_accuracy /= len(dataloader)
            print(f"Final test accuracy: {self.test_accuracy}.")

    def train_step(self, x: torch.Tensor, y: torch.Tensor):
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()

        self.optim.zero_grad()

        prediction = self.model.forward(x)
        target = y

        loss = self.criterion(prediction, target)
        loss.backward()

        self.optim.step()

        loss = loss.detach().cpu().item()
        pred = prediction.detach().cpu().numpy()
        pred = np.argmax(pred, axis=1)
        y = y.detach().cpu().numpy()
        y = np.argmax(y, axis=1)
        acc = accuracy_score(pred, y)
        # acc = (pred == y).mean()
        return loss, acc

    def test_step(self, x: torch.Tensor, y: torch.Tensor):
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()

        prediction = self.model.forward(x)

        pred = prediction.detach().cpu().numpy()
        # pred = np.where(pred > 0.5, 1, 0)
        y = y.detach().cpu().numpy()
        pred = np.argmax(pred, axis=1)
        y = np.argmax(y, axis=1)
        # acc = (pred == y).mean()
        acc = accuracy_score(pred, y)
        return acc

    def plot_loss(self, title):
        now = datetime.now()
        t = now.strftime("%Y-%m-%d")
        save_name = 'output/' + title + t + '.png'
        plt.plot(self.train_loss)
        plt.title(str(title))
        plt.savefig(save_name)
        # plt.plot(self.train_accuracies)
        plt.show()

    def save_data(self, title):
        now = datetime.now()
        t = now.strftime("%Y-%m-%d")
        title = 'output/' + title + t + '.json'
        out_dict = {
            'epoch_accuracies': self.train_accuracies,
            'epoch_loss': self.train_loss,
            'step_accuracies': self.train_acc_steps,
            'step_loss': self.train_loss_steps,
            'avg_test_acc': self.test_accuracy,
        }

        with open(title, 'w') as f:
            json.dump(out_dict, f, indent=4)


