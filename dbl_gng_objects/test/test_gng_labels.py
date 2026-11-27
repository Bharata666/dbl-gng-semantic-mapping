import json

import pytest
import rclpy
from std_msgs.msg import String
from visualization_msgs.msg import Marker

from dbl_gng_objects.gng_objects_node import GngObjectsNode
from test_gng_node_persist import Capture, feed, wait_trained


def start(extra=()):
    args = ['--ros-args', '-p', 'train_min_points:=40', '-p', 'train_interval_s:=0.0', '-p', 'epochs:=3', '-p', 'nice:=0']
    for e in extra:
        args += ['-p', e]
    rclpy.init(args=args)
    n = GngObjectsNode(); n.pub_markers, n.pub_cloud = Capture(), Capture()
    return n


def labels(n, action=Marker.ADD):
    return [m for m in n.pub_markers.msgs[-1].markers if m.ns == 'gng_labels' and m.action == action]


@pytest.mark.parametrize('cls,sent,expected', [(1, 'desk', 'desk'), (0, None, 'chair'), (0, 'kursi', 'kursi')])
def test_each_network_has_a_name_label_above_it(cls, sent, expected):
    n = start()
    try:
        feed(n, (1.0, 0.5, 0.6), cls=cls)
        obj = {'global_id': 1, 'class_id': cls, 'centroid_map': [1.0, 0.5, 0.6], 'bbox_map': {'min_bound': [0.75, 0.25, 0.35], 'max_bound': [1.25, 0.75, 0.85]}}
        if sent:
            obj['class_name'] = sent
        n._names.clear()
        n.on_map(String(data=json.dumps({'objects': [obj]})))
        assert wait_trained(n)
        n._publish_markers()
        (m,) = labels(n)
        assert (m.color.r, m.color.a) == (pytest.approx(0.1), 1.0)
        assert m.text == expected and m.type == Marker.TEXT_VIEW_FACING and m.scale.z == pytest.approx(0.15)
        assert abs(m.pose.position.x - 1.0) < 0.2 and m.pose.position.z > max(s['nodes'][:, 2].max() for s in n.store.snapshot(0))
        n.store.clear(); n._publish_markers()
        assert not labels(n) and len(labels(n, Marker.DELETE)) == 1            # label ikut dihapus bersama jaringannya
    finally:
        n.destroy_node(); rclpy.shutdown()


def test_label_size_zero_hides_labels():
    n = start(['label_size:=0.0'])
    try:
        feed(n, (1.0, 0.5, 0.6))
        assert wait_trained(n)
        n._publish_markers()
        assert not labels(n) and any(m.ns == 'gng_nodes' for m in n.pub_markers.msgs[-1].markers)
    finally:
        n.destroy_node(); rclpy.shutdown()
