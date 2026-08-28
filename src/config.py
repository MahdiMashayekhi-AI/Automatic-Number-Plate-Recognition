CHAR_LIST = ['-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
             'ا', 'ب', 'پ', 'ت', 'ث', 'ج', 'د', 'ز', 'ژ', 'س', 'ش',
             'ص', 'ط', 'ع', 'ف', 'ق', 'ک', 'گ', 'ل', 'م', 'ن', 'و',
             'ه', 'ی']

PLATE_PATTERN = ['d', 'd', 'l', 'd', 'd', 'd', 'd', 'd']

DIGIT_SET = {ch for ch in CHAR_LIST if ch.isdigit()}
LETTER_SET = {ch for ch in CHAR_LIST if ch != '-' and ch not in DIGIT_SET}

CHAR2IDX = {char: idx for idx, char in enumerate(CHAR_LIST)}
IDX2CHAR = {idx: char for idx, char in enumerate(CHAR_LIST)}

MIN_VOTES_TO_CONFIRM = 3

BLUR_THRESHOLD = 30

MIN_QUAD_AREA = 500
MIN_ASPECT_RATIO = 1
MAX_ASPECT_RATIO = 8

IMAGE_WIDTH = 128
IMAGE_HEIGHT = 32
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50