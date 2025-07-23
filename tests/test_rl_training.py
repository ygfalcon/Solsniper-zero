from solhunter_zero.rl_training import RLTraining
from solhunter_zero.offline_data import OfflineData


def test_rl_training_runs(tmp_path):
    db = f"sqlite:///{tmp_path/'data.db'}"
    data = OfflineData(db)
    data.log_snapshot("tok", 1.0, 1.0, total_depth=1.5, imbalance=0.0, slippage=0.0, volume=0.0)
    data.log_snapshot("tok", 1.1, 1.0, total_depth=1.6, imbalance=0.0, slippage=0.0, volume=0.0)
    data.log_trade("tok", "buy", 1.0, 1.0)
    data.log_trade("tok", "sell", 1.0, 1.1)

    model_path = tmp_path / "ppo_model.pt"
    trainer = RLTraining(db_url=db, model_path=model_path)
    trainer.train()
    assert model_path.exists()
