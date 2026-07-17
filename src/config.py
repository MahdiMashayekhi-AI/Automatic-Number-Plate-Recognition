CHAR_LIST = ['-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
             'ا', 'ب', 'پ', 'ت', 'ث', 'ج', 'د', 'ز', 'ژ', 'س', 'ش',
             'ص', 'ط', 'ع', 'ف', 'ق', 'ک', 'گ', 'ل', 'م', 'ن', 'و',
             'ه', 'ی']

CHAR2IDX = {char: idx for idx, char in enumerate(CHAR_LIST)}
IDX2CHAR = {idx: char for idx, char in enumerate(CHAR_LIST)}

IMAGE_WIDTH = 128
IMAGE_HEIGHT = 32
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 30