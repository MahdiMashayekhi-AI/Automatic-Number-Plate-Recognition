LETTER_MAPPING = {'ا': 'A', 'ب': 'B', 'پ': 'P', 'ت': 'T', 'ث': 'S', 'ج': 'J', 'د': 'D', 'ز': 'Z', 'ژ': 'ZH', 'س': 'SIN', 'ش': 'SH', 'ص': 'SAD', 'ط': 'TA', 'ع': 'E', 'ف': 'F', 'ق': 'GH', 'ک': 'K', 'گ': 'G', 'ل': 'L', 'م': 'M', 'ن': 'N', 'و': 'V', 'ه': 'H', 'ی': 'Y'}

def change_fa_to_en(text: str) -> str:
  if not text or not isinstance(text, str):
    return text

  for fa, en in LETTER_MAPPING.items():
    if fa in text:
      text = text.replace(fa, en)

  return text