ALL_WRITERS = [
    'ange', 'fola', 'gina', 'ibra', 'mae', 'mayo', 'nick',
    'odosa', 'sam', 'scott', 'sipo', 'som', 'steve'
]

WRITER_TO_ID = {writer: idx for idx, writer in enumerate(ALL_WRITERS)}
ID_TO_WRITER = {idx: writer for idx, writer in enumerate(ALL_WRITERS)}

# Training parameters
BATCH_SIZE = 8
LEARNING_RATE = 0.001
NUM_EPOCHS = 30
IMAGE_SIZE = 224
