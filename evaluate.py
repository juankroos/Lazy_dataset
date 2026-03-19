"""
evaluate.py — Évaluation proxy d'une structure de dataset candidate

Modes :
  1. Proxy RandomForest (défaut)
  2. Mode A  --model-desc "CNN, image 224x224, 10 classes, pas de GPU"
  3. Mode B1 --model-path model.py [--epochs 5]

Usage :
    python evaluate.py schema.json
    python evaluate.py schema.json --model-desc "CNN, image 224x224, 10 classes"
    python evaluate.py schema.json --model-path model.py --epochs 5 --verbose
    python evaluate.py schema.json --seeds 5 --all-schemas history.json
"""

import json, time, argparse, importlib.util, re, sys
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

N_SAMPLES   = 300
N_CV_FOLDS  = 5
TARGET_TIME = 10.0


# ══════════════════════════════════════════════════════════════════════════════
# MODE A — PROFIL EXTRAIT DE LA DESCRIPTION
# ══════════════════════════════════════════════════════════════════════════════

class ModelProfile:
    """Profil extrait d'une description textuelle. Oriente données et fitness."""

    def __init__(self):
        self.arch            = "unknown"
        self.has_cnn         = False
        self.has_rnn         = False
        self.has_transformer = False
        self.image_size      = (32, 32)
        self.n_channels      = 1
        self.sequence_len    = None
        self.n_classes       = 4
        self.task            = "classification"
        self.has_gpu         = False
        self.batch_size      = 32
        self.latency_ms      = None
        self.embedded        = False
        self.fitness_weights = {"accuracy":0.50,"simplicity":0.25,"novelty":0.15,"speed":0.10}

    @classmethod
    def from_description(cls, desc: str) -> "ModelProfile":
        p   = cls()
        txt = desc.lower()

        # architecture
        if any(k in txt for k in ("cnn","convolution","conv","resnet","vgg","efficientnet")):
            p.has_cnn = True; p.arch = "cnn"
        if any(k in txt for k in ("lstm","gru","rnn","recurrent")):
            p.has_rnn = True
            p.arch = "rnn" if p.arch == "unknown" else "hybrid"
        if any(k in txt for k in ("transformer","attention","bert","vit","gpt")):
            p.has_transformer = True
            p.arch = "transformer" if p.arch == "unknown" else "hybrid"
        if p.arch == "unknown":
            p.arch = "mlp"

        # taille image
        m = re.search(r"(\d{2,4})\s*[x×]\s*(\d{2,4})", txt)
        if m:
            p.image_size = (int(m.group(1)), int(m.group(2)))
        else:
            m2 = re.search(r"image[s]?\s+(\d{2,4})", txt)
            if m2:
                sz = int(m2.group(1)); p.image_size = (sz, sz)

        # canaux
        if any(k in txt for k in ("rgb","3 channel","3 canal","couleur","color")):
            p.n_channels = 3

        # classes
        m = re.search(r"(\d+)\s*class", txt) or re.search(r"(\d+)\s*catégorie", txt)
        if m: p.n_classes = int(m.group(1))

        # tâche
        if any(k in txt for k in ("régression","regression","valeur continue","score continu")):
            p.task = "regression"

        # séquence
        m = re.search(r"(?:sequence|séquence|window|fenêtre)[^\d]*(\d+)", txt)
        if m: p.sequence_len = int(m.group(1))
        elif p.has_rnn: p.sequence_len = 10

        # hardware
        if any(k in txt for k in ("gpu","cuda")): p.has_gpu = True
        if any(k in txt for k in ("raspberry","embarqué","embedded","edge","mobile")):
            p.embedded = True

        m = re.search(r"batch[^\d]*(\d+)", txt)
        if m: p.batch_size = int(m.group(1))

        m = re.search(r"(\d+)\s*ms", txt)
        if m: p.latency_ms = int(m.group(1))

        # ajuster poids fitness
        if p.embedded or (p.latency_ms and p.latency_ms < 100):
            p.fitness_weights = {"accuracy":0.35,"simplicity":0.35,"novelty":0.10,"speed":0.20}
        elif p.task == "regression":
            p.fitness_weights = {"accuracy":0.45,"simplicity":0.25,"novelty":0.20,"speed":0.10}
        elif p.has_transformer:
            p.fitness_weights = {"accuracy":0.55,"simplicity":0.20,"novelty":0.15,"speed":0.10}

        return p

    def summary(self) -> str:
        lines = [
            f"  Architecture  : {self.arch}",
            f"  Image         : {self.image_size[0]}x{self.image_size[1]} ({self.n_channels}ch)",
            f"  Tâche         : {self.task} / {self.n_classes} classes",
            f"  GPU           : {'oui' if self.has_gpu else 'non'}",
            f"  Embarqué      : {'oui' if self.embedded else 'non'}",
        ]
        if self.sequence_len:
            lines.append(f"  Seq. length   : {self.sequence_len}")
        if self.latency_ms:
            lines.append(f"  Latence max   : {self.latency_ms} ms")
        w = self.fitness_weights
        lines.append(
            f"  Poids fitness : acc={w['accuracy']} simp={w['simplicity']} "
            f"nov={w['novelty']} spd={w['speed']}"
        )
        return "\n".join(lines)

    def adapt_field(self, field: dict) -> dict:
        """Adapte un champ schema pour correspondre aux attentes du modèle."""
        field = dict(field)
        ftype = field.get("type","")
        if ftype.startswith("matrix_") and self.has_cnn:
            h, w = self.image_size
            field["type"] = f"matrix_{h}x{w}"
        if self.sequence_len and (
            "history" in field.get("name","") or
            field.get("derivation","").startswith("window")
        ):
            field["_seq_hint"] = self.sequence_len
        return field


# ══════════════════════════════════════════════════════════════════════════════
# GÉNÉRATION DE DONNÉES SYNTHÉTIQUES
# ══════════════════════════════════════════════════════════════════════════════

def generate_field(field: dict, n: int, rng: np.random.Generator,
                   profile: "ModelProfile | None" = None) -> np.ndarray:
    ftype = field.get("type","vector_10")
    deriv = field.get("derivation","raw")

    if ftype.startswith("matrix_"):
        parts = ftype.split("_")[1].split("x")
        r, c  = int(parts[0]), int(parts[1])
        if profile and profile.has_cnn and profile.n_channels > 1:
            data = rng.random((n, profile.n_channels, r, c)).astype(np.float32)
        else:
            data = rng.random((n, r, c)).astype(np.float32)

    elif ftype.startswith("vector_"):
        size = int(ftype.split("_")[1])
        if profile and profile.has_rnn and profile.sequence_len:
            data = rng.random((n, profile.sequence_len, size)).astype(np.float32)
        else:
            data = rng.random((n, size)).astype(np.float32)

    elif ftype.startswith("scalar"):
        data = rng.random((n, 1)).astype(np.float32)

    elif "label" in ftype or "class" in ftype:
        nc = profile.n_classes if profile else 4
        for part in ftype.split("_"):
            if "class" in part:
                try: nc = int(part.replace("classes","").replace("class","") or nc)
                except ValueError: pass
        data = rng.integers(0, nc, size=(n,1))
    else:
        data = rng.random((n, 10)).astype(np.float32)

    if deriv == "raw": return data
    if deriv in ("mean","image_mean"):
        return data.mean(axis=tuple(range(1,data.ndim))).reshape(n,1)
    if deriv in ("stats","temp_stats"):
        flat = data.reshape(n,-1)
        return np.column_stack([flat.mean(1),flat.std(1),flat.min(1),flat.max(1)])
    if deriv.startswith("window") or deriv.startswith("temp_history"):
        win = profile.sequence_len if profile and profile.sequence_len else 5
        for tok in re.findall(r"\d+", deriv):
            win = int(tok); break
        flat = data.reshape(n,-1)
        return np.stack([np.roll(flat,i,axis=1) for i in range(win)],axis=2).reshape(n,-1)
    if deriv.startswith("concat"):
        return data.reshape(n,-1)
    if deriv in ("normalized","image_normalized"):
        flat = data.reshape(n,-1)
        mn,mx = flat.min(axis=1,keepdims=True), flat.max(axis=1,keepdims=True)
        return (flat-mn)/(mx-mn+1e-8)
    if deriv.startswith("corr"):
        return rng.random((n,1)).astype(np.float32)
    if deriv.startswith("pca"):
        flat = data.reshape(n,-1); return flat[:,:min(8,flat.shape[1])]
    if deriv.startswith("fft"):
        flat = data.reshape(n,-1)
        return np.abs(np.fft.rfft(flat,axis=1))[:,:min(10,flat.shape[1])].astype(np.float32)
    return data.reshape(n,-1)


def build_dataset(schema: dict, n: int, seed: int,
                  profile: "ModelProfile | None" = None):
    rng          = np.random.default_rng(seed)
    input_parts  = []
    output_field = None

    fields = [profile.adapt_field(f) for f in schema["fields"]] if profile else schema["fields"]

    for field in fields:
        if field["role"] == "output":
            output_field = field
        elif field["role"] == "input":
            arr = generate_field(field, n, rng, profile)
            input_parts.append(arr.reshape(n,-1))

    if not input_parts:  raise ValueError("Aucun champ input trouvé.")
    if output_field is None: raise ValueError("Aucun champ output trouvé.")

    X = np.hstack(input_parts)

    signal = X[:,0]
    sig_n  = (signal - signal.min()) / ((signal.max()-signal.min()) + 1e-8)
    noise  = rng.random(n)
    task   = profile.task if profile else "classification"
    nc     = profile.n_classes if profile else 4

    if task == "regression":
        y = (0.6*sig_n + 0.4*noise).astype(np.float32)
    else:
        ftype = output_field.get("type","text_label")
        for part in ftype.split("_"):
            if "class" in part:
                try: nc = int(part.replace("classes","").replace("class","") or nc)
                except ValueError: pass
        y = ((0.6*sig_n + 0.4*noise)*nc).astype(int).clip(0, nc-1)

    return X, y


# ══════════════════════════════════════════════════════════════════════════════
# MODE B1 — PYTORCH + ADAPTATEUR DE FORMAT
# ══════════════════════════════════════════════════════════════════════════════

def load_pytorch_model(model_path: str):
    """
    Charge un nn.Module depuis un fichier .py.

    Le fichier doit exposer au minimum :
      build_model(input_dim, n_classes) -> nn.Module

    Il peut aussi exposer (optionnel) :
      INPUT_SHAPE  = (C, H, W)        # format attendu par sample (sans batch dim)
      INPUT_SHAPES = {"image": (C,H,W), "temp": (T, F)}  # un shape par champ nommé

    Exemples dans model.py :
      # MLP simple — aucun INPUT_SHAPE nécessaire, le vecteur aplati convient
      def build_model(input_dim, n_classes): ...

      # CNN — déclarer INPUT_SHAPE pour que l'adaptateur reshape correctement
      INPUT_SHAPE = (3, 224, 224)
      def build_model(input_dim, n_classes): ...

      # Modèle multi-entrées — déclarer INPUT_SHAPES par nom de champ
      INPUT_SHAPES = {"image": (3, 224, 224), "temp": (1, 10)}
      def build_model(input_dim, n_classes): ...
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch non installé. Lancer: pip install torch")

    spec = importlib.util.spec_from_file_location("user_model", model_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if not (hasattr(mod, "build_model") or hasattr(mod, "Model")):
        raise ValueError(
            f"{model_path} doit définir build_model(input_dim, n_classes) "
            f"ou une classe Model(input_dim, n_classes)."
        )

    factory      = mod.build_model if hasattr(mod, "build_model") else mod.Model
    input_shape  = getattr(mod, "INPUT_SHAPE",  None)   # (C, H, W) ou (T, F)
    input_shapes = getattr(mod, "INPUT_SHAPES", None)   # {"field_name": shape, ...}

    return factory, torch, input_shape, input_shapes


class ModelAdapter:
    """
    Adapte les données synthétiques (N, flat_dim) au format attendu par le modèle.

    Stratégie de détection (par ordre de priorité) :
      1. INPUT_SHAPES déclaré dans model.py  → reshape champ par champ puis concat
      2. INPUT_SHAPE déclaré dans model.py   → reshape tout X vers (N, *INPUT_SHAPE)
      3. Profil CNN détecté (via model-desc) → reshape vers (N, C, H, W)
      4. Profil RNN détecté                  → reshape vers (N, T, F)
      5. Forward pass de test sur X aplati   → si ça passe, garder tel quel (MLP)
      6. Fallback                            → garder X aplati, logger un warning
    """

    def __init__(self, model, torch, profile, input_shape, input_shapes,
                 schema, flat_dim, verbose=False):
        self.torch         = torch
        self.profile       = profile
        self.input_shape   = input_shape   # depuis model.py
        self.input_shapes  = input_shapes  # depuis model.py
        self.schema        = schema
        self.flat_dim      = flat_dim
        self.verbose       = verbose
        self.detected_mode = "flat"        # flat | single_reshape | field_reshape | rnn
        self.target_shape  = None          # shape par sample (sans batch dim)

        self._detect(model)

    def _detect(self, model):
        torch = self.torch

        # ── priorité 1 : INPUT_SHAPES par champ ──
        if self.input_shapes:
            self.detected_mode = "field_reshape"
            if self.verbose:
                print(f"  Adaptateur : field_reshape via INPUT_SHAPES={self.input_shapes}")
            return

        # ── priorité 2 : INPUT_SHAPE global ──
        if self.input_shape:
            self.detected_mode = "single_reshape"
            self.target_shape  = self.input_shape
            if self.verbose:
                print(f"  Adaptateur : single_reshape vers {self.input_shape}")
            return

        # ── priorité 3 : profil CNN ──
        if self.profile and self.profile.has_cnn:
            c = self.profile.n_channels
            h, w = self.profile.image_size
            self.detected_mode = "single_reshape"
            self.target_shape  = (c, h, w)
            if self.verbose:
                print(f"  Adaptateur : CNN détecté → reshape vers ({c},{h},{w})")
            return

        # ── priorité 4 : profil RNN ──
        if self.profile and self.profile.has_rnn and self.profile.sequence_len:
            t = self.profile.sequence_len
            f = max(1, self.flat_dim // t)
            self.detected_mode = "single_reshape"
            self.target_shape  = (t, f)
            if self.verbose:
                print(f"  Adaptateur : RNN détecté → reshape vers ({t},{f})")
            return

        # ── priorité 5 : forward pass test avec X aplati ──
        try:
            dummy = torch.zeros(2, self.flat_dim)
            model.eval()
            with torch.no_grad():
                model(dummy)
            self.detected_mode = "flat"
            if self.verbose:
                print(f"  Adaptateur : MLP/flat — forward pass OK ({self.flat_dim} dims)")
        except Exception as e:
            # forward a planté → on ne sait pas le format, fallback flat + warning
            self.detected_mode = "flat"
            if self.verbose:
                print(f"  Adaptateur : forward test échoué ({e}), fallback flat")

    def reshape(self, X_batch):
        """Transforme un batch numpy (N, flat) en tensor au bon format."""
        torch = self.torch
        n     = X_batch.shape[0]

        if self.detected_mode == "single_reshape" and self.target_shape:
            needed = 1
            for d in self.target_shape: needed *= d
            if X_batch.shape[1] >= needed:
                return torch.FloatTensor(
                    X_batch[:, :needed].reshape(n, *self.target_shape)
                )
            else:
                # trop peu de dims → pad avec des zéros
                pad = np.zeros((n, needed - X_batch.shape[1]), dtype=np.float32)
                X_pad = np.hstack([X_batch, pad])
                return torch.FloatTensor(X_pad.reshape(n, *self.target_shape))

        if self.detected_mode == "field_reshape":
            # reconstruire chaque champ et concat selon INPUT_SHAPES
            parts = []
            cursor = 0
            input_fields = [f for f in self.schema["fields"] if f["role"] == "input"]
            for field in input_fields:
                fname = field["name"]
                shape = self.input_shapes.get(fname)
                if shape:
                    size = 1
                    for d in shape: size *= d
                    chunk = X_batch[:, cursor:cursor+size]
                    if chunk.shape[1] < size:
                        pad   = np.zeros((n, size-chunk.shape[1]), dtype=np.float32)
                        chunk = np.hstack([chunk, pad])
                    parts.append(chunk.reshape(n, *shape))
                    cursor += size
                else:
                    # champ sans shape déclaré → garder aplati
                    fsize = X_batch.shape[1] - cursor
                    parts.append(X_batch[:, cursor:cursor+fsize])
                    cursor += fsize
            # concat tout sur dim=1 après reshape plat
            flat_parts = [p.reshape(n,-1) for p in parts]
            return torch.FloatTensor(np.hstack(flat_parts))

        # flat (MLP ou fallback)
        return torch.FloatTensor(X_batch)


def eval_pytorch(factory, torch_mod, input_shape, input_shapes,
                 X, y, schema, profile, epochs=5, seed=0, verbose=False):
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    torch_mod.manual_seed(seed)
    task      = profile.task if profile else "classification"
    n_classes = profile.n_classes if profile else int(y.max()) + 1
    bs        = profile.batch_size if profile else 32

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=seed)

    # ── instancier le modèle ──
    try:    model = factory(X.shape[1], n_classes)
    except: model = factory(X.shape[1])

    device = torch_mod.device("cuda" if torch_mod.cuda.is_available() else "cpu")
    model  = model.to(device)

    # ── construire l'adaptateur ──
    adapter = ModelAdapter(
        model=model, torch=torch_mod, profile=profile,
        input_shape=input_shape, input_shapes=input_shapes,
        schema=schema, flat_dim=X.shape[1], verbose=verbose
    )

    # ── préparer targets ──
    if task == "classification":
        y_tr_t  = torch_mod.LongTensor(y_tr)
        crit    = nn.CrossEntropyLoss()
    else:
        y_tr_t  = torch_mod.FloatTensor(y_tr).unsqueeze(1)
        crit    = nn.MSELoss()

    opt = optim.Adam(model.parameters(), lr=1e-3)

    # DataLoader avec numpy arrays — on reshape dans la boucle via adapter
    from torch.utils.data import TensorDataset, DataLoader
    loader = DataLoader(
        TensorDataset(torch_mod.FloatTensor(X_tr), y_tr_t),
        batch_size=bs, shuffle=True
    )

    t0 = time.perf_counter()
    model.train()
    for ep in range(epochs):
        total = 0.0
        for xb_flat, yb in loader:
            # adapter reshape xb_flat → format attendu par le modèle
            xb = adapter.reshape(xb_flat.numpy()).to(device)
            yb = yb.to(device)
            opt.zero_grad()
            try:
                loss = crit(model(xb), yb)
                loss.backward()
                opt.step()
                total += loss.item()
            except Exception as e:
                if verbose:
                    print(f"    forward error ep{ep}: {e}")
                break
        if verbose:
            print(f"    epoch {ep+1}/{epochs}  loss={total/max(len(loader),1):.4f}")
    elapsed = time.perf_counter() - t0

    # ── évaluation ──
    model.eval()
    with torch_mod.no_grad():
        xv = adapter.reshape(X_val).to(device)
        try:
            out = model(xv)
            if task == "classification":
                preds = out.argmax(1).cpu().numpy()
                score = float((preds == y_val).mean())
            else:
                preds  = out.squeeze().cpu().numpy()
                ss_res = ((y_val - preds) ** 2).sum()
                ss_tot = ((y_val - y_val.mean()) ** 2).sum()
                score  = float(max(0.0, 1.0 - ss_res / (ss_tot + 1e-8)))
        except Exception as e:
            if verbose: print(f"    eval error: {e}")
            score = 0.0

    return score, elapsed


# ══════════════════════════════════════════════════════════════════════════════
# MÉTRIQUES
# ══════════════════════════════════════════════════════════════════════════════

def compute_simplicity(schema):
    ni = len([f for f in schema["fields"] if f["role"]=="input"])
    nd = len([f for f in schema["fields"]
              if f.get("derivation","raw")!="raw" and f["role"]=="input"])
    return 1.0/(1.0+ni+0.5*nd)

def compute_novelty(schema, all_schemas):
    if not all_schemas: return 0.5
    def sig(s):
        return "|".join(f"{f['name']}:{f.get('derivation','raw')}"
                        for f in sorted(s["fields"],key=lambda x:x["name"]))
    cur = sig(schema)
    dists = []
    for other in all_schemas:
        a,b = set(cur.split("|")), set(sig(other).split("|"))
        dists.append(len(a.symmetric_difference(b))/max(len(a|b),1))
    return min(float(np.mean(dists)),1.0)

def compute_robustness(scores):
    return float(np.std(scores)) if len(scores)>1 else 0.0

def estimate_memory_mb(schema, n=10_000):
    total=0
    for f in schema["fields"]:
        ftype=f.get("type","vector_10")
        if ftype.startswith("matrix_"):
            p=ftype.split("_")[1].split("x"); total+=int(p[0])*int(p[1])
        elif ftype.startswith("vector_"): total+=int(ftype.split("_")[1])
        else: total+=1
    return (total*n*4)/(1024**2)

def is_explicable(schema):
    derivs=[f.get("derivation","raw") for f in schema["fields"] if f["role"]=="input"]
    has_stats  = any(d in ("stats","temp_stats","mean","image_mean") for d in derivs)
    has_complex= any(d in ("pca","fft","concat") for d in derivs)
    n_fields   = len([f for f in schema["fields"] if f["role"]=="input"])
    if has_complex or n_fields>6: return 0
    if has_stats and n_fields<=4: return 2
    return 1


# ══════════════════════════════════════════════════════════════════════════════
# ÉVALUATION PRINCIPALE
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(schema, seeds, all_schemas, profile=None,
             model_path=None, epochs=5, verbose=False):

    accuracy_scores, train_times = [], []
    use_pytorch = model_path is not None
    factory = torch_mod = input_shape = input_shapes = None

    if use_pytorch:
        try:
            factory, torch_mod, input_shape, input_shapes = load_pytorch_model(model_path)
            if verbose:
                print(f"  Modèle PyTorch chargé : {model_path}")
                if input_shape:  print(f"  INPUT_SHAPE  : {input_shape}")
                if input_shapes: print(f"  INPUT_SHAPES : {input_shapes}")
        except Exception as e:
            print(json.dumps({"error": str(e)})); sys.exit(1)

    task = profile.task if profile else "classification"

    for seed in seeds:
        try:
            X, y = build_dataset(schema, N_SAMPLES, seed, profile)

            if use_pytorch:
                score, elapsed = eval_pytorch(
                    factory, torch_mod, input_shape, input_shapes,
                    X, y, schema, profile,
                    epochs=epochs, seed=seed, verbose=verbose
                )
            else:
                t0 = time.perf_counter()
                if task == "classification":
                    le = LabelEncoder(); y = le.fit_transform(y)
                    mdl = RandomForestClassifier(n_estimators=50,max_depth=6,
                                                 random_state=seed,n_jobs=-1)
                    sc  = cross_val_score(mdl,X,y,cv=N_CV_FOLDS,scoring="accuracy")
                else:
                    mdl = RandomForestRegressor(n_estimators=50,max_depth=6,
                                                random_state=seed,n_jobs=-1)
                    sc  = cross_val_score(mdl,X,y,cv=N_CV_FOLDS,scoring="r2")
                elapsed = time.perf_counter()-t0
                score   = float(np.mean(sc))

            accuracy_scores.append(score)
            train_times.append(elapsed)
            if verbose:
                print(f"  seed={seed}  score={score:.3f}  t={elapsed:.2f}s")

        except Exception as e:
            if verbose: print(f"  seed={seed}  ERREUR: {e}")
            accuracy_scores.append(0.0)
            train_times.append(TARGET_TIME)

    accuracy   = max(0.0, min(1.0, float(np.mean(accuracy_scores))))
    avg_time   = float(np.mean(train_times))
    simplicity = compute_simplicity(schema)
    novelty    = compute_novelty(schema, all_schemas)
    speed      = 1.0/(1.0+avg_time)
    robustness = compute_robustness(accuracy_scores)

    w = profile.fitness_weights if profile else \
        {"accuracy":0.50,"simplicity":0.25,"novelty":0.15,"speed":0.10}

    fitness = (w["accuracy"]*accuracy + w["simplicity"]*simplicity +
               w["novelty"]*novelty   + w["speed"]*speed)

    result = {
        "schema_id"      : schema.get("schema_id","unknown"),
        "eval_mode"      : "pytorch" if use_pytorch else ("mode_a" if profile else "proxy_rf"),
        "fitness"        : round(fitness,4),
        "accuracy"       : round(accuracy,4),
        "simplicity"     : round(simplicity,4),
        "novelty"        : round(novelty,4),
        "speed"          : round(speed,4),
        "robustness"     : round(robustness,4),
        "memory_mb"      : round(estimate_memory_mb(schema),2),
        "explicable"     : is_explicable(schema),
        "train_time_s"   : round(avg_time,2),
        "fitness_weights": w,
        "n_seeds"        : len(seeds),
        "n_fields"       : len([f for f in schema["fields"] if f["role"]=="input"]),
    }

    if profile:
        result["model_profile"] = {
            "arch"      : profile.arch,
            "task"      : profile.task,
            "image_size": f"{profile.image_size[0]}x{profile.image_size[1]}",
            "n_channels": profile.n_channels,
            "n_classes" : profile.n_classes,
            "gpu"       : profile.has_gpu,
            "embedded"  : profile.embedded,
        }

    return result


# ══════════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Évalue une structure de dataset candidate.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
exemples :
  python evaluate.py schema.json
  python evaluate.py schema.json --model-desc "CNN, image 224x224, 10 classes, pas GPU"
  python evaluate.py schema.json --model-path model.py --epochs 10 --verbose
        """
    )
    parser.add_argument("schema")
    parser.add_argument("--seeds",       type=int, default=3)
    parser.add_argument("--all-schemas", default=None,
                        help="Fichier JSON ou string JSON array de schémas précédents")
    parser.add_argument("--verbose",     action="store_true")
    parser.add_argument("--model-desc",  default=None, metavar="DESC",
                        help="Description texte du modèle (Mode A)")
    parser.add_argument("--model-path",  default=None, metavar="PY",
                        help="Fichier model.py PyTorch (Mode B1)")
    parser.add_argument("--epochs",      type=int, default=5,
                        help="Époques d'entraînement PyTorch (défaut: 5)")
    args = parser.parse_args()

    path = Path(args.schema)
    if not path.exists():
        print(json.dumps({"error": f"Fichier introuvable : {args.schema}"})); sys.exit(1)
    with open(path) as f:
        schema = json.load(f)

    all_schemas = []
    if args.all_schemas:
        p = Path(args.all_schemas)
        try:
            src = open(p).read() if p.exists() else args.all_schemas
            all_schemas = json.loads(src)
        except Exception:
            pass

    profile = ModelProfile.from_description(args.model_desc) if args.model_desc else None

    if args.verbose:
        mode = "PyTorch" if args.model_path else ("Mode A" if profile else "Proxy RF")
        print(f"\nÉvaluation : {schema.get('schema_id','?')}")
        print(f"Mode       : {mode}")
        if profile:
            print("Profil détecté :")
            print(profile.summary())
        print(f"Champs     : {[f['name'] for f in schema['fields'] if f['role']=='input']}")
        print(f"Seeds      : {list(range(args.seeds))}\n")

    result = evaluate(
        schema=schema, seeds=list(range(args.seeds)),
        all_schemas=all_schemas, profile=profile,
        model_path=args.model_path, epochs=args.epochs,
        verbose=args.verbose,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
