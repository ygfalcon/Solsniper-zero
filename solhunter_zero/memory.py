from __future__ import annotations
import datetime
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    Float,
    String,
    DateTime,
    Text,
)
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

def utcnow():
    return datetime.datetime.utcnow()

class Trade(Base):
    __tablename__ = 'trades'
    id = Column(Integer, primary_key=True)
    token = Column(String, nullable=False)
    direction = Column(String, nullable=False)
    amount = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=utcnow)
    reason = Column(Text)


class VaRLog(Base):
    __tablename__ = "var_logs"

    id = Column(Integer, primary_key=True)
    value = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=utcnow)


class Memory:
    def __init__(self, url: str = 'sqlite:///memory.db'):
        self.engine = create_engine(url, echo=False, future=True)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)

    def log_trade(self, **kwargs) -> None:
        with self.Session() as session:
            trade = Trade(**kwargs)
            session.add(trade)
            session.commit()

    def log_var(self, value: float) -> None:
        """Record a value-at-risk measurement."""
        with self.Session() as session:
            rec = VaRLog(value=value)
            session.add(rec)
            session.commit()

    def list_trades(self):
        with self.Session() as session:
            return session.query(Trade).all()

    def list_vars(self):
        with self.Session() as session:
            return session.query(VaRLog).all()
