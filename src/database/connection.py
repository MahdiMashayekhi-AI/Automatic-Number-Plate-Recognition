from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base


try:
  engine = create_engine('sqlite:///./anpr.db')
  print("Connection created successfuly.")
except Exception as e:
  print("Connection could not be made due to the following error:\n", e)


session = sessionmaker(autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
  db = session()
  try:
    yield db
  finally:
    db.close()