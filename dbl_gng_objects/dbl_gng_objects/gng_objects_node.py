"""Node representasi DBL-GNG per objek untuk peta semantik.

Membaca (tanpa mengubah apa pun):
  /semantic_map/global_objects_pointcloud  titik objek per frame di frame `map` (class_id)
  /semantic_map/global_semantic_map        objek peta (id, kelas, kotak)
  /semantic_map/reset_map_cmd              kosongkan peta (GNG ikut dikosongkan; bila persist_file diisi, GNG DIPERTAHANKAN dan diarsipkan)
  /semantic_map/gng_clear_saved            hapus semua GNG (termasuk yang tersimpan di berkas)
Menerbitkan:
  /semantic_map/gng_points   (PointCloud2, xyzrgb)  titik objek terkumpul, berwarna kelas
  /semantic_map/gng_markers  (MarkerArray)          node (SPHERE_LIST) dan edge (LINE_LIST) DBL-GNG per objek, berwarna kelas

DBL-GNG: dbl_gng_objects/dbl_gng_core.py (salinan tanpa perubahan dari proyek dbl_gng_pca, MIT, Mathematics 12(12):1909, 2024).
"""

import json
import os
import threading
import time
from typing import Any, Dict, List

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Empty, Header, String
from visualization_msgs.msg import Marker, MarkerArray
import sensor_msgs_py.point_cloud2 as pc2

from dbl_gng_objects.gng_objects import GngObjectStore
from dbl_gng_objects.yolo_palette import get_rgb, get_rgba_float


def _pack_rgb(r: int, g: int, b: int) -> float:
    return float(np.array([(int(r) << 16) | (int(g) << 8) | int(b)], dtype=np.uint32).view(np.float32)[0])


class GngObjectsNode(Node):
    def __init__(self) -> None:
        super().__init__('gng_objects_node')
        self.declare_parameter('points_topic', '/semantic_map/global_objects_pointcloud')
        self.declare_parameter('map_topic', '/semantic_map/global_semantic_map')
        self.declare_parameter('reset_topic', '/semantic_map/reset_map_cmd')
        self.declare_parameter('target_frame', 'map')
        self.declare_parameter('voxel', 0.02)
        self.declare_parameter('max_points', 1500)
        self.declare_parameter('train_min_points', 50)
        self.declare_parameter('train_interval_s', 2.0)
        self.declare_parameter('epochs', 40)
        self.declare_parameter('margin', 0.15)
        self.declare_parameter('display_points', 300)       # titik per objek yang diterbitkan ke RViz (bentuk ada di node/edge)
        self.declare_parameter('marker_refresh_s', 1.0)     # terbitkan ulang marker walau tak berubah (untuk RViz yang baru dibuka)
        self.declare_parameter('cloud_min_period_s', 3.0)   # jeda minimum antar awan titik
        self.declare_parameter('cloud_refresh_s', 6.0)
        self.declare_parameter('nice', 10)                  # prioritas CPU lebih rendah daripada YOLO
        self.declare_parameter('node_size', 0.04)
        self.declare_parameter('edge_width', 0.008)
        # Penyimpanan antar sesi: kosong = mati (perilaku lama persis). Diisi = GNG (titik, node, edge) disimpan ke berkas .npz dan dimuat lagi saat start.
        self.declare_parameter('persist_file', '')
        # keep_unseen: GNG tetap tampil selama sesi walau objeknya keluar pandangan (false = perilaku lama: dihapus setelah 5 dtk hilang dari peta)
        self.declare_parameter('keep_unseen', False)
        self.declare_parameter('save_interval_s', 10.0)
        self.declare_parameter('supersede_dist', 0.5)       # arsip diganti bila objek kelas sama terdeteksi lagi dalam jarak ini
        self.declare_parameter('clear_saved_topic', '/semantic_map/gng_clear_saved')

        g = lambda k: self.get_parameter(k).value
        try:
            os.nice(int(g('nice')))
        except (OSError, ValueError):
            pass
        self.frame = str(g('target_frame'))
        self.margin = float(g('margin'))
        self.display_points = int(g('display_points'))
        self.node_size, self.edge_width = float(g('node_size')), float(g('edge_width'))
        self.persist_file = os.path.expanduser(str(g('persist_file')))
        self.store = GngObjectStore(voxel=float(g('voxel')), max_points=int(g('max_points')), train_min_points=int(g('train_min_points')),
                                    train_interval_s=float(g('train_interval_s')), epochs=int(g('epochs')),
                                    persist=bool(self.persist_file), supersede_dist=float(g('supersede_dist')), keep_unseen=bool(g('keep_unseen')))
        self._last_saved_version = -1
        self._last_save = time.time()
        if self.persist_file:
            n, note = self.store.load(self.persist_file, time.time())
            self._last_saved_version = self.store.version
            if note:
                self.get_logger().warn(note)
            elif n:
                self.get_logger().info(f'GNG tersimpan dimuat: {n} objek dari {self.persist_file}')

        self.pub_markers = self.create_publisher(MarkerArray, '/semantic_map/gng_markers', 10)
        # reliable: pesan awan titik cukup besar dan jarang; best-effort sering hilang di WiFi yang sibuk
        self.pub_cloud = self.create_publisher(PointCloud2, '/semantic_map/gng_points', QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE))
        self.create_subscription(PointCloud2, str(g('points_topic')), self.on_points, qos_profile_sensor_data)
        self.create_subscription(String, str(g('map_topic')), self.on_map, QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE))
        self.create_subscription(Empty, str(g('reset_topic')), self.on_reset, 10)
        self.create_subscription(Empty, str(g('clear_saved_topic')), self.on_clear_saved, 10)

        self._published_ids: set = set()
        self._last_version = -1
        self._last_shape = -1
        self._last_marker_pub = 0.0
        self._last_cloud_pub = 0.0
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._train_loop, daemon=True)
        self._worker.start()
        self.create_timer(0.25, self.on_timer)
        self.get_logger().info('GNG objek aktif (DBL-GNG per objek: node + edge + titik, frame %s)' % self.frame)

    # ---------------------------------------------------------------------------------- masukan
    def on_map(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except Exception:
            return
        objs = []
        for o in data.get('objects', []):
            try:
                b = o['bbox_map']
                objs.append({'id': int(o['global_id']), 'class_id': int(o['class_id']), 'bmin': b['min_bound'], 'bmax': b['max_bound'],
                             'centroid': o['centroid_map']})
            except (KeyError, TypeError, ValueError):
                continue
        self.store.update_objects(objs, time.time())

    def on_points(self, msg: PointCloud2) -> None:
        if msg.width * msg.height == 0:
            return
        try:
            arr = pc2.read_points_numpy(msg, field_names=['x', 'y', 'z', 'class_id'])
        except Exception:
            return
        if len(arr) == 0:
            return
        self.store.add_points(arr[:, :3].astype(np.float64), np.rint(arr[:, 3]).astype(np.int64), time.time(), self.margin)

    def on_reset(self, _msg: Empty) -> None:
        if self.persist_file:                            # peta objek dikosongkan, tetapi GNG tersimpan dipertahankan (diarsipkan)
            self.store.new_epoch()
            self._save(force=True)
            self._publish(force=True)
            self.get_logger().info('Peta objek dikosongkan; GNG tersimpan dipertahankan')
            return
        self.store.clear()
        self._publish(force=True)
        self.get_logger().info('GNG objek dikosongkan')

    def on_clear_saved(self, _msg: Empty) -> None:
        self.store.clear()
        if self.persist_file:
            try:
                os.remove(self.persist_file)
            except FileNotFoundError:
                pass
            except OSError as e:
                self.get_logger().warn(f'Gagal menghapus berkas GNG tersimpan: {e}')
            self._last_saved_version = self.store.version
        self._publish(force=True)
        self.get_logger().info('Semua GNG dihapus (termasuk yang tersimpan)')

    def _save(self, force: bool = False) -> None:
        if not self.persist_file:
            return
        now = time.time()
        if not force and (self.store.version == self._last_saved_version or now - self._last_save < float(self.get_parameter('save_interval_s').value)):
            return
        err = self.store.save(self.persist_file)
        self._last_save = now
        if err:
            self.get_logger().warn(f'Gagal menyimpan GNG: {err}')
        else:
            self._last_saved_version = self.store.version

    # --------------------------------------------------------------------------------- pelatihan
    def _train_loop(self) -> None:
        while not self._stop.is_set():
            now = time.time()
            oid = self.store.next_to_train(now)
            if oid is None:
                self._stop.wait(0.25)
                continue
            t0 = time.time()
            ok = self.store.train_object(oid, time.time())
            if ok:
                self.get_logger().debug(f'DBL-GNG objek #{oid} dilatih dalam {time.time() - t0:.1f} s')

    # ------------------------------------------------------------------------------- penerbitan
    def on_timer(self) -> None:
        now = time.time()
        shape_changed = self.store.shape_version != self._last_shape
        if shape_changed or now - self._last_marker_pub >= float(self.get_parameter('marker_refresh_s').value):
            self._publish_markers()
        cloud_due = (self.store.version != self._last_version and now - self._last_cloud_pub >= float(self.get_parameter('cloud_min_period_s').value)) \
            or now - self._last_cloud_pub >= float(self.get_parameter('cloud_refresh_s').value)
        if cloud_due:
            self._publish_cloud()
        self._save()

    def _publish(self, force: bool = False) -> None:
        self._publish_markers()
        self._publish_cloud()

    def _publish_markers(self) -> None:
        snap = self.store.snapshot(0)
        self._last_shape, self._last_marker_pub = self.store.shape_version, time.time()
        header = Header(); header.stamp = self.get_clock().now().to_msg(); header.frame_id = self.frame

        arr = MarkerArray()
        ids = {s['id'] for s in snap}
        for oid in self._published_ids - ids:          # objek yang sudah hilang dari peta
            for ns in ('gng_nodes', 'gng_edges'):
                d = Marker(); d.header = header; d.ns = ns; d.id = oid; d.action = Marker.DELETE; arr.markers.append(d)
        self._published_ids = ids

        for s in snap:
            r, gcol, b, _ = get_rgba_float(s['class_id'])
            if len(s['nodes']):
                m = Marker(); m.header = header; m.ns = 'gng_nodes'; m.id = s['id']; m.type = Marker.SPHERE_LIST; m.action = Marker.ADD
                m.scale.x = m.scale.y = m.scale.z = self.node_size
                m.color.r, m.color.g, m.color.b, m.color.a = r, gcol, b, 1.0
                m.pose.orientation.w = 1.0
                m.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in s['nodes']]
                arr.markers.append(m)
                if len(s['edges']):
                    e = Marker(); e.header = header; e.ns = 'gng_edges'; e.id = s['id']; e.type = Marker.LINE_LIST; e.action = Marker.ADD
                    e.scale.x = self.edge_width
                    e.color.r, e.color.g, e.color.b, e.color.a = min(1.0, r * 1.1 + 0.1), min(1.0, gcol * 1.1 + 0.1), min(1.0, b * 1.1 + 0.1), 0.95
                    e.pose.orientation.w = 1.0
                    pts = []
                    for a_, b_ in s['edges']:
                        pa, pb = s['nodes'][a_], s['nodes'][b_]
                        pts.append(Point(x=float(pa[0]), y=float(pa[1]), z=float(pa[2])))
                        pts.append(Point(x=float(pb[0]), y=float(pb[1]), z=float(pb[2])))
                    e.points = pts
                    arr.markers.append(e)
        self.pub_markers.publish(arr)

    def _publish_cloud(self) -> None:
        snap = self.store.snapshot(self.display_points)
        self._last_version, self._last_cloud_pub = self.store.version, time.time()
        header = Header(); header.stamp = self.get_clock().now().to_msg(); header.frame_id = self.frame
        pts_xyz: List[np.ndarray] = []
        pts_rgb: List[np.ndarray] = []
        pts_cid: List[np.ndarray] = []
        for s in snap:
            if len(s['points']):
                rr, gg, bb = get_rgb(s['class_id'])
                pts_xyz.append(s['points']); pts_rgb.append(np.full(len(s['points']), _pack_rgb(rr, gg, bb), dtype=np.float32))
                pts_cid.append(np.full(len(s['points']), float(s['class_id']), dtype=np.float32))
        fields = [PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1), PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                  PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1), PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
                  PointField(name='class_id', offset=16, datatype=PointField.FLOAT32, count=1)]      # class_id: dibaca GUI (RViz mengabaikannya)
        data = np.column_stack([np.vstack(pts_xyz).astype(np.float32), np.concatenate(pts_rgb), np.concatenate(pts_cid)]) if pts_xyz else np.empty((0, 5), dtype=np.float32)
        self.pub_cloud.publish(pc2.create_cloud(header, fields, data))

    def destroy_node(self):
        self._stop.set()
        try:
            self._save(force=True)
        except Exception:
            pass
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GngObjectsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
