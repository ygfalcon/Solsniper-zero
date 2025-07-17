from __future__ import annotations
import datetime
from typing import Iterable, List

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


def utcnow() -> datetime.datetime:
    return datetime.datetime.utcnow()


class Trade(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True)
    token = Column(String, nullable=False)
    direction = Column(String, nullable=False)
    amount = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=utcnow)
    reason = Column(Text)


class SimulationLog(Base):
    __tablename__ = "simulation_logs"

    id = Column(Integer, primary_key=True)
    token = Column(String, nullable=False)
    success = Column(Float, nullable=False)
    roi = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=utcnow)


class Memory:
    """SQL-backed storage for trades and simulation history."""

    def __init__(self, url: str = "sqlite:///memory.db"):
        self.engine = create_engine(url, echo=False, future=True)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)

    # ------------------------------------------------------------------
    # Schema upgrade
    # ------------------------------------------------------------------
    def upgrade_schema(self, ddl_statements: Iterable[str]) -> None:
        """Run DDL statements to modify the schema on the fly."""

        with self.engine.begin() as conn:
            for stmt in ddl_statements:
                conn.exec_driver_sql(stmt)

    # ------------------------------------------------------------------
    # Trade logging
    # ------------------------------------------------------------------
    def log_trade(self, *, token: str, direction: str, amount: float, price: float, reason: str | None = None) -> None:
        with self.Session() as session:
            trade = Trade(token=token, direction=direction, amount=amount, price=price, reason=reason)
            session.add(trade)
            session.commit()

    def list_trades(self) -> List[Trade]:
        with self.Session() as session:
            return list(session.query(Trade).order_by(Trade.timestamp.desc()).all())

    # ------------------------------------------------------------------
    # Simulation logging
    # ------------------------------------------------------------------
    def log_simulations(self, token: str, sims: Iterable[SimulationLog | dict]) -> None:
        """Persist a collection of simulation results."""

        with self.Session() as session:
            for result in sims:
                if isinstance(result, dict):
                    entry = SimulationLog(token=token, **result)
                else:
                    entry = result
                    entry.token = token
                session.add(entry)
            session.commit()

    def recent_simulations(self, token: str, limit: int = 100) -> List[SimulationLog]:
        with self.Session() as session:
            q = (
                session.query(SimulationLog)
                .filter_by(token=token)
                .order_by(SimulationLog.timestamp.desc())
                .limit(limit)
            )
            return list(q.all())
