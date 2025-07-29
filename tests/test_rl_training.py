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


def test_dl_workers_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DL_WORKERS", "4")
    captured = {}

    import solhunter_zero.rl_training as rl_training

    def dummy_loader(*a, **k):
        captured["workers"] = k.get("num_workers")
        class L:
            def __iter__(self):
                return iter([])
        return L()

    monkeypatch.setattr(rl_training, "DataLoader", dummy_loader)
    def fake_fit(self, model, data):
        data.setup()
        data.train_dataloader()

    monkeypatch.setattr(rl_training.pl.Trainer, "fit", fake_fit)
    monkeypatch.setattr(rl_training.torch, "save", lambda *a, **k: None)

    db = f"sqlite:///{tmp_path/'data.db'}"
    data = rl_training.OfflineData(db)
    data.log_snapshot("tok", 1.0, 1.0, total_depth=1.0, imbalance=0.0)
    data.log_trade("tok", "buy", 1.0, 1.0)

    trainer = rl_training.RLTraining(db_url=db, model_path=tmp_path/'m.pt')
    trainer.train()

    assert captured.get("workers") == 4
