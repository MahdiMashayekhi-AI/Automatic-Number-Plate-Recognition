import logging


def setup_logger():
  logger = logging.getLogger()
  logger.setLevel(logging.DEBUG)

  if logger.hasHandlers():
    logger.handlers.clear()

  formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

  console_handler = logging.StreamHandler()
  console_handler.setLevel(logging.INFO)
  console_handler.setFormatter(formatter)

  file_handler = logging.FileHandler("anpr.log")
  file_handler.setLevel(logging.DEBUG)
  file_handler.setFormatter(formatter)

  logger.addHandler(console_handler)
  logger.addHandler(file_handler)

  return logger