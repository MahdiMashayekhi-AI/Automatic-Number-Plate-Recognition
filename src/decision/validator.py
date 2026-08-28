from src.config import PLATE_PATTERN, DIGIT_SET, LETTER_SET


def is_valid_plate_format(text):
  if len(text) != 8:
    return False

  for ch, expected_type in zip(text, PLATE_PATTERN):
    if expected_type == 'd' and ch not in DIGIT_SET:
      return False

    if expected_type == 'l' and ch not in LETTER_SET:
      return False

  return True

