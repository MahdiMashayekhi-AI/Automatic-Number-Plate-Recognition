import logging
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base


logger = logging.getLogger(__name__)

try:
  engine = create_engine('sqlite:///./anpr.db')
  logging.info("Connection created successfuly.")
except Exception as e:
  logging.error("Connection could not be made due to the following error:\n", e)


session = sessionmaker(autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
  db = session()
  try:
    yield db
  finally:
    db.close()


@contextmanager
def get_db_context():
  db = session()
  try:
    yield db
    db.commit()
  except Exception as e:
    db.rollback()
    logger.error(f"Database operation failed: {e}")
    raise
  finally:
    db.close()