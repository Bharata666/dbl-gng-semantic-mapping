"""Logika murni (tanpa ROS) representasi DBL-GNG per objek untuk ltc_mapping.

Titik tiap objek (frame `map`) dikumpulkan dari beberapa pandangan, dikurangi per voxel, lalu dilatih DBL-GNG
(inti `dbl_gng_core.py`, salinan tanpa perubahan dari proyek dbl_gng_pca: https://github.com/CornerSiow/DBL-GNG, MIT,
Mathematics 12(12):1909, 2024) menjadi node + edge. Parameter pelatihan sama dengan object_store.py di dbl_gng_pca:
M = clip(N/15, 20, 200), K = clip(M/8, 3, 12), 40 epoch.

Modul ini hanya MEMBACA data ltc_mapping; tidak mengubah peta objek, YOLO, atau point cloud.
"""

import os
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from dbl_gng_objects.dbl_gng_core import DBL_GNG


def voxel_merge(P: np.ndarray, voxel: float, cap: int, rng: np.random.Generator) -> np.ndarray:
    """Satu titik per voxel (yang tertua), lalu batasi jumlah total dengan subsampel acak."""
    if len(P) == 0:
        return P
    key = np.floor(P / voxel).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    Q = P[np.sort(idx)]
    if len(Q) > cap:
        Q = Q[np.sort(rng.choice(len(Q), cap, replace=False))]
    return Q


def edge_pairs(C) -> np.ndarray:
    pairs = {tuple(sorted((int(a), int(b)))) for a, b in C if a != b}
    return np.array(sorted(pairs), dtype=np.int32).reshape(-1, 2)


def assign_points(points: np.ndarray, class_ids: np.ndarray, objects: List[Dict[str, Any]], margin: float = 0.15) -> np.ndarray:
    """Id objek peta untuk tiap titik (-1 = tidak ada). Titik milik objek bila kelasnya sama dan berada di dalam
    kotak objek yang diperlebar `margin`; bila beberapa objek memenuhi, dipilih yang pusatnya terdekat."""
    out = np.full(len(points), -1, dtype=np.int64)
    best = np.full(len(points), np.inf)
    for o in objects:
        lo = np.asarray(o['bmin'], float) - margin
        hi = np.asarray(o['bmax'], float) + margin
        m = (class_ids == o['class_id']) & np.all((points >= lo) & (points <= hi), axis=1)
        if not m.any():
            continue
        d = np.linalg.norm(points[m] - np.asarray(o['centroid'], float), axis=1)
        sel = np.nonzero(m)[0]
        better = d < best[sel]
        out[sel[better]] = o['id']
        best[sel[better]] = d[better]
    return out


def train_gng(X: np.ndarray, epochs: int = 40) -> Tuple[np.ndarray, np.ndarray]:
    """Latih DBL-GNG pada titik X (N x 3) -> (node W, edge E). Parameter sama dengan object_store.py."""
    X = np.asarray(X, dtype=np.float64)
    M = int(np.clip(round(len(X) / 15), 20, 200))
    K = int(np.clip(M // 8, 3, 12))
    net = DBL_GNG(3, M)
    net.initializeDistributedNode(X, K)
    for _ in range(epochs):
        net.resetBatch()
        net.batchLearning(X)
        net.updateNetwork()
        net.addNewNode()
    return np.asarray(net.W[:, :3], dtype=np.float64), edge_pairs(net.C)


KEYMUL = 1_000_000          # kunci entri = epoch * KEYMUL + id objek peta (hanya bila persist=True)


class GngObjectStore:
    """Titik, node, dan edge DBL-GNG per objek peta (aman untuk dipakai dari beberapa thread).

    persist=False (bawaan): perilaku lama persis. persist=True: GNG tidak ikut terhapus saat peta objek dikosongkan, ltc dinyalakan ulang, atau objeknya keluar pandangan/hilang dari daftar peta; entri sesi
    lama diarsipkan (tetap tampil dan bisa disimpan ke berkas), dan diganti otomatis bila objek kelas sama terdeteksi lagi di dekatnya.
    """

    def __init__(self, voxel: float = 0.02, max_points: int = 2500, train_min_points: int = 80,
                 train_interval_s: float = 8.0, grace_s: float = 5.0, epochs: int = 40, seed: int = 0,
                 persist: bool = False, supersede_dist: float = 0.5, keep_unseen: bool = False):
        self.voxel, self.max_points, self.train_min_points = voxel, max_points, train_min_points
        self.train_interval_s, self.grace_s, self.epochs = train_interval_s, grace_s, epochs
        self.persist, self.supersede_dist, self.epoch = bool(persist), float(supersede_dist), 0
        # keep_unseen: GNG tetap tampil walau objeknya keluar pandangan/hilang dari daftar peta (selama sesi). persist (antar sesi) juga mengaktifkannya.
        self.keep_unseen = bool(keep_unseen) or self.persist
        self.geom: Dict[int, Dict[str, Any]] = {}
        self.objects: List[Dict[str, Any]] = []
        self._lock = threading.RLock()
        self._rng = np.random.default_rng(seed)
        self.version = 0         # naik tiap titik objek berubah (untuk awan titik)
        self.shape_version = 0   # naik hanya bila node/edge berubah, objek hilang, atau dikosongkan (untuk marker)

    # -- peta objek -----------------------------------------------------------------------------------
    def _key(self, oid: int) -> int:
        return self.epoch * KEYMUL + int(oid) if self.persist else int(oid)

    def update_objects(self, objects: List[Dict[str, Any]], now: float) -> None:
        with self._lock:
            if self.persist:
                objects = [{**o, 'id': self._key(o['id'])} for o in objects]
            self.objects = objects
            present = {o['id']: o for o in objects}
            for oid, o in present.items():
                g = self.geom.get(oid)
                if g is not None and g['class_id'] != o['class_id']:
                    del self.geom[oid]; self.version += 1; self.shape_version += 1
                    g = None
                if g is not None:
                    g['last_seen'] = now
            if not self.keep_unseen:              # perilaku lama: GNG objek yang hilang dari peta dihapus setelah masa tenggang
                for oid in [k for k, g in self.geom.items() if k not in present and now - g['last_seen'] > self.grace_s]:
                    del self.geom[oid]
                    self.version += 1; self.shape_version += 1
            else:
                self._supersede(present, now)

    def _supersede(self, present: Dict[int, Dict[str, Any]], now: float) -> None:
        """Mode persist: GNG TIDAK dihapus hanya karena objeknya keluar pandangan atau hilang dari daftar peta (tetap tampil). Entri yang tidak ada di peta
        (sesi lama, atau objek yang hilang > grace_s) hanya diganti bila objek kelas sama yang ada di peta dan sudah punya titik berada dalam supersede_dist
        (data baru menggantikan; juga kasus duplikat yang digabung)."""
        live = [g for k, g in self.geom.items() if k in present and len(g['points']) >= max(10, self.train_min_points // 2)]
        if not live:
            return
        stale = [k for k, g in self.geom.items() if k not in present and (k // KEYMUL < self.epoch or now - g['last_seen'] > self.grace_s)]
        for k in stale:
            a = self.geom[k]
            if not len(a['points']):
                continue
            ca = a['points'].mean(axis=0)
            if any(g['class_id'] == a['class_id'] and np.linalg.norm(g['points'].mean(axis=0) - ca) < self.supersede_dist for g in live):
                del self.geom[k]; self.version += 1; self.shape_version += 1

    def new_epoch(self) -> None:
        """Peta objek dikosongkan: persist=True mengarsipkan GNG yang ada (id peta mulai dari 1 lagi tanpa bentrok); persist=False menghapusnya (perilaku lama)."""
        with self._lock:
            if not self.persist:
                self.geom.clear(); self.objects = []
            else:
                self.epoch += 1; self.objects = []
            self.version += 1; self.shape_version += 1

    def clear(self) -> None:
        with self._lock:
            self.geom.clear(); self.objects = []; self.version += 1; self.shape_version += 1

    # -- titik ---------------------------------------------------------------------------------------
    def add_points(self, points: np.ndarray, class_ids: np.ndarray, now: float, margin: float = 0.15) -> int:
        """Tambahkan titik pandangan terbaru ke objek yang memilikinya. Mengembalikan jumlah titik yang dipakai."""
        with self._lock:
            objs = list(self.objects)
        if not objs or len(points) == 0:
            return 0
        owner = assign_points(points, class_ids, objs, margin)
        used = 0
        with self._lock:
            by_id = {o['id']: o for o in self.objects}
            for oid in np.unique(owner[owner >= 0]):
                o = by_id.get(int(oid))
                if o is None:
                    continue
                pts = points[owner == oid]
                g = self.geom.setdefault(int(oid), {'class_id': o['class_id'], 'points': np.empty((0, 3)), 'nodes': np.empty((0, 3)),
                                                    'edges': np.empty((0, 2), dtype=np.int32), 'trained_n': 0,
                                                    't_trained': -1e9, 'last_seen': now, 'dirty': False})
                before = len(g['points'])
                g['points'] = voxel_merge(np.vstack([g['points'], pts]) if before else pts, self.voxel, self.max_points, self._rng)
                g['last_seen'] = now
                if len(g['points']) != before:
                    g['dirty'] = True; self.version += 1
                used += len(pts)
        return used

    # -- pelatihan --------------------------------------------------------------------------------------
    def next_to_train(self, now: float) -> Optional[int]:
        with self._lock:
            for oid, g in self.geom.items():
                n = len(g['points'])
                if n >= self.train_min_points and now - g['t_trained'] >= self.train_interval_s and \
                        (len(g['nodes']) == 0 or n >= 1.15 * g['trained_n'] + 20):
                    return oid
        return None

    def train_object(self, oid: int, now: float) -> bool:
        with self._lock:
            g = self.geom.get(oid)
            if g is None or len(g['points']) < self.train_min_points:
                return False
            X = g['points'].copy()
        try:
            W, E = train_gng(X, self.epochs)
        except (IndexError, ValueError):          # inti tidak punya pengaman untuk data buruk; pertahankan node lama
            with self._lock:
                if oid in self.geom:
                    self.geom[oid]['t_trained'] = now
            return False
        with self._lock:
            g = self.geom.get(oid)
            if g is None:
                return False
            g['nodes'], g['edges'], g['trained_n'], g['t_trained'], g['dirty'] = W, E, len(X), now, False
            self.version += 1; self.shape_version += 1
        return True

    # -- penyimpanan antar sesi ---------------------------------------------------------------------------
    def save(self, path: str) -> Optional[str]:
        """Simpan semua entri (titik, node, edge) secara atomik. Mengembalikan pesan galat atau None."""
        if not path:
            return None
        with self._lock:
            items = [(k, g) for k, g in self.geom.items()]
            data: Dict[str, Any] = {'keys': np.array([k for k, _ in items], dtype=np.int64), 'classes': np.array([g['class_id'] for _, g in items], dtype=np.int64),
                                    'trained_n': np.array([g['trained_n'] for _, g in items], dtype=np.int64), 'epoch': np.array([self.epoch], dtype=np.int64)}
            for i, (_, g) in enumerate(items):
                data[f'p{i}'], data[f'n{i}'], data[f'e{i}'] = g['points'], g['nodes'], g['edges']
        tmp = path + '.tmp.npz'
        try:
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
            np.savez_compressed(tmp, **data)
            os.replace(tmp, path)
            return None
        except OSError as e:
            return str(e)

    def load(self, path: str, now: float) -> Tuple[int, str]:
        """Muat entri tersimpan sebagai arsip (epoch baru = epoch tersimpan + 1). Berkas hilang/rusak -> (0, catatan); tidak pernah melempar."""
        if not path or not os.path.exists(path):
            return 0, ''
        try:
            with np.load(path) as z:
                keys, classes, trained = z['keys'], z['classes'], z['trained_n']
                saved_epoch = int(z['epoch'][0])
                loaded: Dict[int, Dict[str, Any]] = {}
                for i, k in enumerate(keys):
                    P, W, E = np.asarray(z[f'p{i}'], float), np.asarray(z[f'n{i}'], float), np.asarray(z[f'e{i}'], np.int32)
                    if P.ndim != 2 or P.shape[1] != 3 or W.ndim != 2 or W.shape[-1] != 3 or E.ndim != 2 or (len(E) and (E.min() < 0 or E.max() >= len(W))):
                        raise ValueError(f'entri {i} tidak valid')
                    loaded[int(k)] = {'class_id': int(classes[i]), 'points': P, 'nodes': W, 'edges': E.reshape(-1, 2), 'trained_n': int(trained[i]),
                                      't_trained': now, 'last_seen': now, 'dirty': False}
        except Exception as e:                    # berkas rusak/terpotong: abaikan, mulai kosong
            return 0, f'berkas GNG tersimpan tidak terbaca ({type(e).__name__}: {e}); mulai kosong'
        with self._lock:
            self.geom.update(loaded)
            self.epoch = max(self.epoch, saved_epoch + 1)
            self.version += 1; self.shape_version += 1
        return len(loaded), ''

    # -- tampilan --------------------------------------------------------------------------------------
    def snapshot(self, max_display_points: int = 1500) -> List[Dict[str, Any]]:
        with self._lock:
            out = []
            for oid, g in self.geom.items():
                P = g['points']
                if len(P) > max_display_points:
                    P = P[np.linspace(0, len(P) - 1, max_display_points).astype(int)]
                out.append({'id': oid, 'class_id': g['class_id'], 'points': P.copy(), 'nodes': g['nodes'].copy(), 'edges': g['edges'].copy()})
            return out
