import pandas as pd


class CONFIG:
    """
    Configuration for training of neural network.
    Learning rate should be between 0.1 and 0.01.
    Batch size between 16 and 128.
    Layer size not smaller than 128
    Layer num not smaller than 6
    """
    # FIXED
    EPOCHS = 10
    INPUT_SIZE = 784
    OUTPUT_CLASSES = 10
    # CHANGE
    LEARNING_RATE = (0.01, 0.1)
    BATCH_SIZE = (16, 128)
    LAYER_SIZE = (16, 64)
    LAYER_NUMBER = (2, 8)

    def select_parameters(self, col_data: pd.Series):
        lr = CONFIG.LEARNING_RATE[int(col_data['learning_rate'])]
        bs = CONFIG.BATCH_SIZE[int(col_data['batch_size'])]
        ls = CONFIG.LAYER_SIZE[int(col_data['layer_size'])]
        ln = CONFIG.LAYER_NUMBER[int(col_data['layer_number'])]
        return lr, bs, ls, ln

    def __str__(self):
        return f"Epochs: {CONFIG.EPOCHS}\n" \
               f"Input size: {CONFIG.INPUT_SIZE}\n" \
               f"Output classes: {CONFIG.OUTPUT_CLASSES}\n" \
               f"Learning rate: {CONFIG.LEARNING_RATE}\n" \
               f"Batch size: {CONFIG.BATCH_SIZE}\n" \
               f"Layer size: {CONFIG.LAYER_SIZE}\n" \
               f"Layer number: {CONFIG.LAYER_NUMBER}\n" \
