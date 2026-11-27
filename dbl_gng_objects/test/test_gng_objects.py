"""Tes logika representasi DBL-GNG per objek (gng_objects.py + inti dbl_gng_core.py)."""
import hashlib
import os

import numpy as np
import pytest

from dbl_gng_objects import dbl_gng_core
from dbl_gng_objects.gng_objects import GngObjectStore, assign_points, edge_pairs, train_gng, voxel_merge


def box_cloud(n, center, size=(0.4, 0.4, 0.5), seed=0):
    rng = np.random.default_rng(seed)
    return np.asarray(center, float) + (rng.uniform(-0.5, 0.5, (n, 3)) * np.asarray(size))


def obj(oid, cls, center, half=0.25):
    c = np.asarray(center, float)
    return {'id': oid, 'class_id': cls, 'bmin': (c - half).tolist(), 'bmax': (c + half).tolist(), 'centroid': c.tolist()}


def test_core_copy_is_identical_to_dbl_gng_pca_if_available():
    src = os.path.expanduser('~/Downloads/dbl_gng_pca/src/dbl_gng/core/dbl_gng.py')
    if not os.path.exists(src):
        pytest.skip('proyek dbl_gng_pca tidak ada di mesin ini')
    h = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
    assert h(src) == h(dbl_gng_core.__file__)


def test_voxel_merge_thins_and_caps():
    rng = np.random.default_rng(0)
    P = np.repeat(box_cloud(300, (0, 0, 0)), 3, axis=0)          # titik ganda
    Q = voxel_merge(P, 0.05, 10_000, rng)
    assert len(Q) < len(P) and len(np.unique(np.floor(Q / 0.05), axis=0)) == len(Q)
    assert len(voxel_merge(P, 0.001, 50, rng)) == 50


def test_assign_points_by_class_and_box():
    objs = [obj(1, 0, (1, 0, 0)), obj(2, 0, (3, 0, 0)), obj(3, 5, (1, 0, 0))]
    P = np.array([[1.0, 0, 0], [3.0, 0, 0], [1.0, 0, 0], [9.0, 9, 9]])
    cls = np.array([0, 0, 5, 0])
    assert assign_points(P, cls, objs, margin=0.1).tolist() == [1, 2, 3, -1]


def test_assign_points_picks_nearest_centroid_when_boxes_overlap():
    objs = [obj(1, 0, (0.0, 0, 0), half=0.5), obj(2, 0, (0.4, 0, 0), half=0.5)]
    assert assign_points(np.array([[0.05, 0, 0], [0.35, 0, 0]]), np.array([0, 0]), objs, 0.0).tolist() == [1, 2]


def test_train_gng_returns_nodes_and_valid_edges():
    X = box_cloud(900, (0, 0, 0))
    W, E = train_gng(X, epochs=15)
    assert W.shape[1] == 3 and 20 <= len(W) <= 200
    assert E.ndim == 2 and E.shape[1] == 2 and len(E) > 0 and E.max() < len(W) and E.min() >= 0
    assert np.all(W.min(0) >= X.min(0) - 0.1) and np.all(W.max(0) <= X.max(0) + 0.1)     # node berada pada objek


def test_edge_pairs_are_unique_and_undirected():
    assert edge_pairs([[1, 2], [2, 1], [3, 3], [0, 4]]).tolist() == [[0, 4], [1, 2]]


def test_store_accumulates_trains_and_publishes_snapshot():
    st = GngObjectStore(voxel=0.02, train_min_points=80, train_interval_s=0.0, epochs=10)
    st.update_objects([obj(7, 0, (1, 1, 0.3))], now=0.0)
    for k in range(3):
        used = st.add_points(box_cloud(400, (1, 1, 0.3), seed=k), np.zeros(400, dtype=int), now=float(k))
        assert used == 400
    oid = st.next_to_train(now=10.0)
    assert oid == 7 and st.train_object(7, now=10.0)
    snap = st.snapshot()
    assert len(snap) == 1 and snap[0]['id'] == 7 and len(snap[0]['nodes']) >= 20 and len(snap[0]['edges']) > 0
    assert st.next_to_train(now=10.1) is None          # belum tumbuh cukup: tidak dilatih ulang


def test_store_ignores_points_of_other_class_or_outside_any_object():
    st = GngObjectStore()
    st.update_objects([obj(1, 0, (0, 0, 0))], now=0.0)
    assert st.add_points(box_cloud(50, (0, 0, 0)), np.full(50, 5), now=0.0) == 0
    assert st.add_points(box_cloud(50, (5, 5, 5)), np.zeros(50, dtype=int), now=0.0) == 0
    assert st.snapshot() == []


def test_store_drops_objects_removed_from_map_after_grace_and_on_class_change_and_on_clear():
    st = GngObjectStore(grace_s=5.0)
    st.update_objects([obj(1, 0, (0, 0, 0)), obj(2, 1, (3, 0, 0))], now=0.0)
    st.add_points(box_cloud(60, (0, 0, 0)), np.zeros(60, dtype=int), now=0.0)
    st.add_points(box_cloud(60, (3, 0, 0)), np.ones(60, dtype=int), now=0.0)
    assert {s['id'] for s in st.snapshot()} == {1, 2}
    st.update_objects([obj(2, 1, (3, 0, 0))], now=3.0)
    assert {s['id'] for s in st.snapshot()} == {1, 2}            # masih dalam masa tenggang
    st.update_objects([obj(2, 1, (3, 0, 0))], now=9.0)
    assert {s['id'] for s in st.snapshot()} == {2}               # objek 1 hilang dari peta
    st.update_objects([obj(2, 5, (3, 0, 0))], now=9.5)
    assert st.snapshot() == []                                   # kelas berubah: geometri lama dibuang
    st.update_objects([obj(2, 5, (3, 0, 0))], now=10.0)
    st.add_points(box_cloud(60, (3, 0, 0)), np.full(60, 5), now=10.0)
    st.clear()
    assert st.snapshot() == []


def test_train_failure_on_degenerate_data_keeps_previous_nodes():
    st = GngObjectStore(train_min_points=5, epochs=3)
    st.update_objects([obj(1, 0, (0, 0, 0), half=1.0)], now=0.0)
    st.add_points(np.zeros((10, 3)), np.zeros(10, dtype=int), now=0.0)      # semua titik sama
    st.train_object(1, now=1.0)                                             # tidak boleh melempar galat
    assert 1 in {s['id'] for s in st.snapshot()}


def test_shape_version_changes_only_on_training_removal_or_clear_not_on_new_points():
    st = GngObjectStore(train_min_points=80, train_interval_s=0.0, epochs=5, grace_s=1.0)
    st.update_objects([obj(1, 0, (0, 0, 0))], now=0.0)
    s0 = st.shape_version
    st.add_points(box_cloud(300, (0, 0, 0)), np.zeros(300, dtype=int), now=0.0)
    assert st.version > 0 and st.shape_version == s0                 # titik baru: awan berubah, bentuk belum
    assert st.train_object(1, now=1.0) and st.shape_version > s0     # pelatihan: bentuk berubah
    s1 = st.shape_version
    st.add_points(box_cloud(20, (0, 0, 0), seed=5), np.zeros(20, dtype=int), now=2.0)
    assert st.shape_version == s1
    st.update_objects([], now=10.0)                                  # objek hilang dari peta
    assert st.shape_version > s1
    s2 = st.shape_version; st.clear()
    assert st.shape_version > s2


def test_snapshot_without_display_points_returns_empty_point_arrays_but_keeps_shape():
    st = GngObjectStore(train_min_points=80, train_interval_s=0.0, epochs=5)
    st.update_objects([obj(1, 0, (0, 0, 0))], now=0.0)
    st.add_points(box_cloud(300, (0, 0, 0)), np.zeros(300, dtype=int), now=0.0)
    st.train_object(1, now=1.0)
    snap = st.snapshot(0)
    assert len(snap[0]['points']) == 0 and len(snap[0]['nodes']) > 0 and len(snap[0]['edges']) > 0


# ------------------------------------------------------------------------------- persist (GNG tersimpan antar sesi)
def trained_store(persist, center=(1.0, 0.0, 0.5), cls=0, oid=1, seed=0, **kw):
    st = GngObjectStore(voxel=0.02, train_min_points=80, train_interval_s=0.0, epochs=5, persist=persist, **kw)
    st.update_objects([obj(oid, cls, center)], 0.0)
    st.add_points(box_cloud(400, center, seed=seed), np.full(400, cls), 0.0)
    assert st.train_object(st._key(oid), 1.0)
    return st


def test_default_is_legacy_new_epoch_clears_everything():
    st = trained_store(False)
    st.new_epoch()
    assert st.snapshot(0) == [] and st.epoch == 0


def test_persist_archives_gng_when_map_is_cleared_and_new_ids_do_not_collide():
    st = trained_store(True)
    old = st.snapshot(0); assert len(old) == 1 and len(old[0]['nodes']) > 0
    st.new_epoch()                                                       # peta objek dikosongkan (mis. RViz dibuka)
    st.update_objects([], 100.0)                                         # jauh melewati grace: arsip tidak boleh terhapus
    assert len(st.snapshot(0)) == 1 and st.epoch == 1
    st.update_objects([obj(1, 0, (4.0, 2.0, 0.5))], 101.0)               # objek baru memakai id 1 lagi
    st.add_points(box_cloud(300, (4.0, 2.0, 0.5), seed=3), np.full(300, 0), 101.0)
    snap = {s['id']: s for s in st.snapshot(0)}
    assert len(snap) == 2 and 1 in snap and 1_000_001 in snap            # entri lama (epoch 0) dan baru (epoch 1) terpisah
    assert np.allclose(snap[1]['nodes'].mean(axis=0), (1.0, 0.0, 0.5), atol=0.3)     # arsip tidak tercampur titik objek baru


def test_archived_entry_is_replaced_when_same_class_object_is_seen_nearby_again():
    st = trained_store(True, center=(1.0, 0.0, 0.5))
    st.new_epoch()
    st.update_objects([obj(1, 0, (1.15, 0.05, 0.5))], 10.0)              # objek yang sama dilihat lagi (sedikit bergeser)
    assert len(st.snapshot(0)) == 1                                       # belum ada titik baru: arsip tetap
    st.add_points(box_cloud(300, (1.15, 0.05, 0.5), seed=5), np.full(300, 0), 10.0)
    st.update_objects([obj(1, 0, (1.15, 0.05, 0.5))], 11.0)
    snap = st.snapshot(0)
    assert len(snap) == 1 and snap[0]['id'] == 1_000_001                  # arsip digantikan data baru (tidak dobel)


def test_archived_entry_stays_when_far_or_other_class():
    st = trained_store(True, center=(1.0, 0.0, 0.5))
    st.new_epoch()
    st.update_objects([obj(1, 0, (3.0, 0.0, 0.5)), obj(2, 5, (1.0, 0.0, 0.5))], 10.0)       # jauh (kelas sama) dan ssama posisi (kelas lain)
    st.add_points(box_cloud(300, (3.0, 0.0, 0.5), seed=6), np.full(300, 0), 10.0)
    st.add_points(box_cloud(300, (1.0, 0.0, 0.5), seed=7), np.full(300, 5), 10.0)
    st.update_objects([obj(1, 0, (3.0, 0.0, 0.5)), obj(2, 5, (1.0, 0.0, 0.5))], 11.0)
    assert len(st.snapshot(0)) == 3


def test_persist_keeps_gng_when_object_leaves_view_or_map_list_but_legacy_still_purges():
    st = trained_store(True)
    st.update_objects([], 100.0)                                          # objek keluar pandangan / hilang dari daftar peta, jauh melewati grace
    snap = st.snapshot(0); assert len(snap) == 1 and len(snap[0]['nodes']) > 0       # GNG tetap tampil
    legacy = trained_store(False)
    legacy.update_objects([], 100.0)
    assert legacy.snapshot(0) == []                                       # perilaku lama tidak berubah


def test_persist_orphan_is_removed_only_when_a_same_class_object_in_the_map_overlaps_it():
    st = GngObjectStore(voxel=0.02, train_min_points=80, train_interval_s=0.0, epochs=5, persist=True)
    st.update_objects([obj(1, 0, (1.0, 0.0, 0.5)), obj(2, 0, (1.3, 0.0, 0.5))], 0.0)       # dua deteksi dekat (nanti digabung)
    st.add_points(box_cloud(300, (1.0, 0.0, 0.5), seed=1), np.full(300, 0), 0.0)
    st.add_points(box_cloud(300, (1.3, 0.0, 0.5), seed=2), np.full(300, 0), 0.0)
    assert len(st.snapshot(0)) == 2
    st.update_objects([obj(1, 0, (1.0, 0.0, 0.5))], 10.0)                 # id 2 digabung ke id 1: hilang dari daftar peta > grace
    assert {s['id'] for s in st.snapshot(0)} == {1}                       # duplikat yang tumpang tindih diganti, tidak dobel
    far = GngObjectStore(voxel=0.02, train_min_points=80, train_interval_s=0.0, epochs=5, persist=True)
    far.update_objects([obj(1, 0, (1.0, 0.0, 0.5)), obj(2, 0, (5.0, 0.0, 0.5))], 0.0)
    far.add_points(box_cloud(300, (1.0, 0.0, 0.5), seed=1), np.full(300, 0), 0.0)
    far.add_points(box_cloud(300, (5.0, 0.0, 0.5), seed=2), np.full(300, 0), 0.0)
    far.update_objects([obj(1, 0, (1.0, 0.0, 0.5))], 10.0)                # id 2 jauh dan hilang: dipertahankan
    assert {s['id'] for s in far.snapshot(0)} == {1, 2}


def test_persist_object_returning_to_the_map_keeps_using_the_same_entry():
    st = trained_store(True, center=(1.0, 0.0, 0.5))
    st.update_objects([], 50.0)
    n0 = len(st.snapshot(0)[0]['points'])
    st.update_objects([obj(1, 0, (1.0, 0.0, 0.5))], 60.0)                 # kembali muncul dengan id yang sama
    st.add_points(box_cloud(300, (1.0, 0.0, 0.5), seed=9), np.full(300, 0), 60.0)
    snap = st.snapshot(0)
    assert len(snap) == 1 and snap[0]['id'] == 1 and len(snap[0]['points']) >= n0


def test_save_load_roundtrip_continues_epoch_and_restores_shapes(tmp_path):
    p = str(tmp_path / 'sub' / 'gng.npz')
    a = trained_store(True, center=(1.0, 0.0, 0.5))
    a.new_epoch(); a.update_objects([obj(1, 5, (2.0, 1.0, 0.8))], 5.0)
    a.add_points(box_cloud(300, (2.0, 1.0, 0.8), seed=2), np.full(300, 5), 5.0); assert a.train_object(a._key(1), 6.0)
    assert a.save(p) is None and not os.path.exists(p + '.tmp.npz')
    b = GngObjectStore(persist=True)
    n, note = b.load(p, 50.0)
    assert n == 2 and note == '' and b.epoch == a.epoch + 1
    sa, sb = {s['id']: s for s in a.snapshot(0)}, {s['id']: s for s in b.snapshot(0)}
    assert set(sa) == set(sb)
    for k in sa:
        assert sa[k]['class_id'] == sb[k]['class_id'] and np.allclose(sa[k]['nodes'], sb[k]['nodes']) and np.array_equal(sa[k]['edges'], sb[k]['edges'])
        assert np.allclose(sa[k]['points'], sb[k]['points'])
    b.update_objects([obj(1, 0, (7.0, 7.0, 0.5))], 51.0)                  # objek baru: id 1 pada epoch baru, tidak menimpa yang dimuat
    assert len({s['id'] for s in b.snapshot(0)} | {b._key(1)}) == 3


def test_load_is_safe_for_missing_corrupt_and_invalid_files(tmp_path):
    st = GngObjectStore(persist=True)
    assert st.load(str(tmp_path / 'tidak_ada.npz'), 0.0) == (0, '') and st.load('', 0.0) == (0, '')
    bad = tmp_path / 'rusak.npz'; bad.write_bytes(b'bukan npz')
    n, note = st.load(str(bad), 0.0); assert n == 0 and 'tidak terbaca' in note and st.snapshot(0) == []
    inv = tmp_path / 'invalid.npz'
    np.savez(inv, keys=np.array([1]), classes=np.array([0]), trained_n=np.array([0]), epoch=np.array([0]),
             p0=np.zeros((5, 2)), n0=np.zeros((3, 3)), e0=np.zeros((0, 2), dtype=np.int32))      # titik bukan N x 3
    n, note = st.load(str(inv), 0.0); assert n == 0 and 'tidak terbaca' in note and st.snapshot(0) == []
    ex = tmp_path / 'edge_di_luar.npz'
    np.savez(ex, keys=np.array([1]), classes=np.array([0]), trained_n=np.array([0]), epoch=np.array([0]),
             p0=np.zeros((5, 3)), n0=np.zeros((3, 3)), e0=np.array([[0, 9]], dtype=np.int32))    # edge menunjuk node yang tidak ada
    n, note = st.load(str(ex), 0.0); assert n == 0 and st.snapshot(0) == []


def test_save_failure_returns_message_and_never_raises(tmp_path):
    st = trained_store(True)
    (tmp_path / 'blok').write_text('x')
    assert st.save(str(tmp_path / 'blok' / 'g.npz')) is not None and st.save('') is None


# ------------------------------------------------------------------------------- keep_unseen (tanpa penyimpanan antar sesi)
def test_keep_unseen_retains_gng_during_session_without_persistence_and_clears_on_reset():
    st = trained_store(False, keep_unseen=True)
    assert st.persist is False and st.keep_unseen is True
    st.update_objects([], 100.0)                                          # objek keluar pandangan / hilang dari daftar peta
    assert len(st.snapshot(0)) == 1                                       # tetap tampil
    st.new_epoch()                                                        # peta objek dikosongkan: GNG ikut kosong (tidak ada antar sesi)
    assert st.snapshot(0) == [] and st.epoch == 0


def test_keep_unseen_off_is_exactly_legacy_and_persist_implies_keep_unseen():
    assert trained_store(False).keep_unseen is False and trained_store(True).keep_unseen is True
    st = trained_store(False)
    st.update_objects([], 100.0); assert st.snapshot(0) == []


def test_keep_unseen_replaces_overlapping_duplicate_but_keeps_distant_orphan():
    st = GngObjectStore(voxel=0.02, train_min_points=80, train_interval_s=0.0, epochs=5, keep_unseen=True)
    st.update_objects([obj(1, 0, (1.0, 0.0, 0.5)), obj(2, 0, (1.3, 0.0, 0.5)), obj(3, 0, (6.0, 0.0, 0.5))], 0.0)
    for i, c in enumerate(((1.0, 0.0, 0.5), (1.3, 0.0, 0.5), (6.0, 0.0, 0.5))):
        st.add_points(box_cloud(300, c, seed=i), np.full(300, 0), 0.0)
    st.update_objects([obj(1, 0, (1.0, 0.0, 0.5))], 10.0)                 # id 2 digabung ke 1; id 3 keluar pandangan
    assert {s['id'] for s in st.snapshot(0)} == {1, 3}
