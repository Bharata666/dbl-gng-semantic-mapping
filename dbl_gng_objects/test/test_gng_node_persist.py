"""Node GNG: penyimpanan antar sesi (persist_file), pengosongan peta, dan penghapusan GNG tersimpan."""
import json
import os
import time

import numpy as np
import pytest
import rclpy
from std_msgs.msg import Empty, Header, String

from dbl_gng_objects.gng_objects_node import GngObjectsNode
from dbl_gng_objects.pointcloud_geometry import build_pointcloud2_msg


class Capture:
    def __init__(self): self.msgs = []
    def publish(self, m): self.msgs.append(m)


def make_node(path):
    rclpy.init(args=['--ros-args', '-p', f'persist_file:={path}', '-p', 'save_interval_s:=0.0', '-p', 'train_min_points:=40', '-p', 'train_interval_s:=0.0',
                     '-p', 'epochs:=3', '-p', 'margin:=0.15', '-p', 'nice:=0'])
    n = GngObjectsNode(); n.pub_markers, n.pub_cloud = Capture(), Capture()
    return n


def close(n):
    n.destroy_node(); rclpy.shutdown()


def feed(n, center, cls=0, oid=1, count=300, seed=0):
    c = np.asarray(center, float); rng = np.random.default_rng(seed)
    obj = {'global_id': oid, 'class_id': cls, 'class_name': 'x', 'centroid_map': c.tolist(), 'bbox_map': {'min_bound': (c - 0.25).tolist(), 'max_bound': (c + 0.25).tolist()}}
    n.on_map(String(data=json.dumps({'objects': [obj]})))
    pts = (c + rng.uniform(-0.2, 0.2, (count, 3))).astype(np.float32)
    h = Header(); h.frame_id = 'map'
    n.on_points(build_pointcloud2_msg(h, pts, np.zeros(count, np.float32), np.zeros(count, np.float32), np.full(count, cls, np.float32)))


def wait_trained(n, timeout=20.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if any(len(s['nodes']) for s in n.store.snapshot(0)):
            return True
        time.sleep(0.1)
    return False


def marker_ns(n):
    last = n.pub_markers.msgs[-1]
    return sorted({(m.ns, m.id) for m in last.markers if m.action == 0})


def test_gng_is_saved_loaded_after_restart_kept_on_map_reset_and_cleared_on_request(tmp_path):
    path = str(tmp_path / 'cfg' / 'gng_saved.npz')
    n = make_node(path)
    feed(n, (1.0, 0.5, 0.6))
    assert wait_trained(n)
    n.on_timer()                                                           # menyimpan (selang simpan 0 dtk)
    assert os.path.exists(path)
    ids_before = {s['id'] for s in n.store.snapshot(0)}
    close(n)

    n2 = make_node(path)                                                   # ltc dinyalakan ulang
    snap = n2.store.snapshot(0)
    assert len(snap) == 1 and len(snap[0]['nodes']) > 0 and snap[0]['id'] in ids_before                 # GNG lama ada lagi
    n2.on_timer(); n2._publish(force=True)
    assert any(ns == 'gng_nodes' for ns, _ in marker_ns(n2))                                              # dan ikut diterbitkan ke RViz

    n2.on_reset(Empty())                                                   # RViz dibuka: peta objek dikosongkan
    assert len(n2.store.snapshot(0)) == 1                                  # GNG tersimpan tetap ada
    feed(n2, (4.0, 2.0, 0.6), oid=1, seed=3)                               # objek baru dengan id peta yang sama (1)
    assert wait_trained(n2)
    assert len(n2.store.snapshot(0)) == 2                                  # tidak bentrok dengan yang lama

    n2.on_clear_saved(Empty())                                             # hapus semua
    assert n2.store.snapshot(0) == [] and not os.path.exists(path)
    close(n2)


def test_without_persist_file_behaviour_is_legacy(tmp_path):
    rclpy.init(args=['--ros-args', '-p', 'train_min_points:=40', '-p', 'train_interval_s:=0.0', '-p', 'epochs:=3', '-p', 'nice:=0'])
    n = GngObjectsNode(); n.pub_markers, n.pub_cloud = Capture(), Capture()
    try:
        assert n.persist_file == '' and n.store.persist is False
        feed(n, (1.0, 0.5, 0.6)); assert wait_trained(n)
        n.on_timer()
        assert list(tmp_path.iterdir()) == []                              # tidak menulis berkas apa pun
        n.on_reset(Empty())
        assert n.store.snapshot(0) == []                                   # dikosongkan bersama peta seperti dulu
    finally:
        close(n)


def test_corrupt_saved_file_does_not_stop_the_node(tmp_path):
    path = str(tmp_path / 'gng_saved.npz'); open(path, 'wb').write(b'rusak')
    n = make_node(path)
    try:
        assert n.store.snapshot(0) == []                                   # mulai kosong
        feed(n, (1.0, 0.5, 0.6)); assert wait_trained(n)
        n.on_timer()
        assert os.path.getsize(path) > 100                                 # berkas rusak ditimpa dengan yang valid
    finally:
        close(n)


def test_node_defaults_keep_unseen_off_and_param_turns_it_on_without_files(tmp_path):
    rclpy.init(args=['--ros-args', '-p', 'train_min_points:=40', '-p', 'train_interval_s:=0.0', '-p', 'epochs:=3', '-p', 'nice:=0'])
    n = GngObjectsNode()
    try:
        assert n.store.keep_unseen is False and n.store.persist is False
    finally:
        close(n)
    rclpy.init(args=['--ros-args', '-p', 'keep_unseen:=true', '-p', 'train_min_points:=40', '-p', 'train_interval_s:=0.0', '-p', 'epochs:=3', '-p', 'nice:=0'])
    n = GngObjectsNode(); n.pub_markers, n.pub_cloud = Capture(), Capture()
    try:
        assert n.store.keep_unseen is True and n.store.persist is False and n.persist_file == ''
        feed(n, (1.0, 0.5, 0.6)); assert wait_trained(n)
        n.on_map(String(data=json.dumps({'objects': []})))                  # objek hilang dari daftar peta
        time.sleep(0.2); n.store.update_objects([], time.time() + 100)
        assert len(n.store.snapshot(0)) == 1                                # GNG tetap ada
        n.on_timer(); assert list(tmp_path.iterdir()) == []                 # tanpa berkas apa pun
        n.on_reset(Empty()); assert n.store.snapshot(0) == []               # peta dikosongkan: GNG ikut kosong
    finally:
        close(n)
