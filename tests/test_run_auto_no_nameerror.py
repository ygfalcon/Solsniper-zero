import sys
import types
import contextlib
import importlib.machinery

_faiss = types.ModuleType('faiss')
_faiss.__spec__ = importlib.machinery.ModuleSpec('faiss', loader=None)
sys.modules.setdefault('faiss', _faiss)
sys.modules.setdefault('sentence_transformers', types.ModuleType('sentence_transformers'))
sys.modules['sentence_transformers'].SentenceTransformer = lambda *a, **k: types.SimpleNamespace(get_sentence_embedding_dimension=lambda:1, encode=lambda x: [])

sklearn = types.ModuleType('sklearn')
sklearn.__spec__ = importlib.machinery.ModuleSpec('sklearn', loader=None)
sys.modules.setdefault('sklearn', sklearn)
sys.modules['sklearn.linear_model'] = types.SimpleNamespace(LinearRegression=object)
sys.modules['sklearn.ensemble'] = types.SimpleNamespace(GradientBoostingRegressor=object, RandomForestRegressor=object)
sys.modules['sklearn.cluster'] = types.SimpleNamespace(KMeans=object, DBSCAN=object)
sys.modules['xgboost'] = types.ModuleType('xgboost')
sys.modules['xgboost'].XGBRegressor = object

torch_mod = types.ModuleType('torch')
torch_mod.__spec__ = importlib.machinery.ModuleSpec('torch', loader=None)
torch_mod.no_grad = contextlib.nullcontext
torch_mod.tensor = lambda *a, **k: None
torch_mod.nn = types.SimpleNamespace(
    LSTM=object,
    Linear=object,
    TransformerEncoder=object,
    TransformerEncoderLayer=object,
    Module=object,
)
torch_mod.optim = types.ModuleType('optim')
torch_mod.optim.__spec__ = importlib.machinery.ModuleSpec('torch.optim', loader=None)
sys.modules['torch'] = torch_mod
torch_mod.utils = types.ModuleType('utils')
torch_mod.utils.data = types.SimpleNamespace(Dataset=object, DataLoader=object)
sys.modules['torch.utils'] = torch_mod.utils
sys.modules['torch.utils.data'] = torch_mod.utils.data
sys.modules['torch.nn'] = torch_mod.nn

_pl_mod = types.ModuleType('pytorch_lightning')
_pl_mod.callbacks = types.SimpleNamespace(Callback=object)
_pl_mod.LightningModule = object
_pl_mod.LightningDataModule = object
_pl_mod.Trainer = object
sys.modules.setdefault('pytorch_lightning', _pl_mod)


import solhunter_zero.config as cfg_mod


def test_run_auto_no_nameerror(monkeypatch, tmp_path):
    with monkeypatch.context() as mp:
        stub_models = types.ModuleType("solhunter_zero.models")
        stub_models.get_model = lambda *a, **k: None
        stub_models.load_compiled_model = lambda *a, **k: None
        stub_models.export_torchscript = lambda *a, **k: None
        mp.setitem(sys.modules, "solhunter_zero.models", stub_models)
        mp.setitem(
            sys.modules,
            "solhunter_zero.models.regime_classifier",
            types.SimpleNamespace(get_model=lambda *a, **k: None),
        )
        stub_models.regime_classifier = sys.modules[
            "solhunter_zero.models.regime_classifier"
        ]
        from solhunter_zero import main as main_module
        cfg_dir = tmp_path / "cfg"
        cfg_dir.mkdir()
        cfg_file = cfg_dir / "cfg.toml"
        cfg_file.write_text("risk_tolerance=0.5")
        (cfg_dir / "active").write_text("cfg.toml")
        monkeypatch.setattr(cfg_mod, "CONFIG_DIR", str(cfg_dir))
        monkeypatch.setattr(cfg_mod, "ACTIVE_CONFIG_FILE", str(cfg_dir / "active"))
        monkeypatch.setattr(main_module, "CONFIG_DIR", str(cfg_dir))

        monkeypatch.setattr(main_module.wallet, "get_active_keypair_name", lambda: "kp")
        monkeypatch.setattr(main_module, "_start_depth_service", lambda cfg: None)
        called = {}
        monkeypatch.setattr(main_module, "main", lambda **kw: called.setdefault("called", True))
        sys.modules["solhunter_zero.data_sync"] = types.SimpleNamespace(sync_recent=lambda: None)
        main_module.data_sync = sys.modules["solhunter_zero.data_sync"]

        main_module.run_auto(iterations=1, dry_run=True)
        assert called.get("called") is True
