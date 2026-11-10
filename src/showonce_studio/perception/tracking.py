"""Thread lifecycle/state of the working image tracker; no robot or depth-pose imports."""
import collections
import threading


class Tracker(threading.Thread):
    def __init__(self, cam, sam, objects, T_base_cam):
        super().__init__(daemon=True)
        self.cam, self.sam, self.objects, self.T = cam, sam, objects, T_base_cam
        self.masks, self.anchor_frame, self.hue, self.z_table = {}, {}, {}, {}
        self.poses, self.lock, self.stop = {}, threading.Lock(), False
        self.updated = threading.Condition(self.lock)
        self.times, self.lost, self.reanchors, self.cross = [], collections.Counter(), collections.Counter(), []
        self.anchor_rejected, self.implausible = collections.Counter(), collections.Counter()
        self.history = collections.OrderedDict()  # frame_id → {id: pos} (SAM 교차 검증용)
        self.anchor_area, self.last_frame, self.last_frame_t = {}, -1, None
        self.seen = collections.defaultdict(lambda: collections.deque(maxlen=90))  # id → (프레임 시각, 위치) 최근 ~3 s
        # 0.1 s 넘게 걸린 프레임의 단계별 시간(ms). 2026-10-24 v10 T06: 한 프레임 처리가 ~0.38 s 걸려 최신 프레임 나이가 0.2 s를 넘고
        # 3주기 연속 stale → 래치. 어느 단계인지 모르므로 기록한다
        self.slow = []

    def run(self):
        last = -1
        while not self.stop:
            f = self.cam.wait_newer(last)
            if f is None or f["id"] == last:
                continue
            last = f["id"]
            self.process(f)
