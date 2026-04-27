import glob
import json
import os
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class ScanRecord:
    ts_path: str
    meta_path: str
    subject: str
    session: str
    task: str
    acquisition: str
    run: str
    site: str
    site_raw: str
    dataset: str
    sex: str
    age: float
    diag: int
    diag_name: str
    tr: float
    n_timepoints: int
    scan_id: str


@dataclass
class WindowRecord:
    record: ScanRecord
    start: int
    end: int
    window_index: int
    n_windows: int


def _scan_stem(ts_path: str):
    return os.path.basename(ts_path).replace("_roi_ts.npy", "")


ABIDE_SITE_NORMALIZATION = {
    "ABIDEII-KKI_1": "KKI",
    "ABIDEII-NYU_1": "NYU",
    "ABIDEII-OHSU_1": "OHSU",
    "ABIDEII-SDSU_1": "SDSU",
    "ABIDEII-TCD_1": "Trinity",
    "ABIDEII-UCLA_1": "UCLA_1",
    "ABIDEII-USM_1": "USM",
}


def _parse_bids_entities(scan_stem: str):
    entities = {}
    for part in scan_stem.split("_"):
        if "-" not in part:
            continue
        key, value = part.split("-", 1)
        entities[key] = value
    return entities


def normalize_site_name(site: str):
    return ABIDE_SITE_NORMALIZATION.get(site, site)


def _read_record(ts_path: str):
    meta_path = ts_path.replace("_roi_ts.npy", "_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)

    scan_stem = _scan_stem(ts_path)
    entities = _parse_bids_entities(scan_stem)
    diag = meta.get("diag", meta.get("dx", -1))
    diag_name = meta.get("diag_name", meta.get("dx_name", "unknown"))
    subject = meta.get("subject", f"sub-{entities['sub']}" if "sub" in entities else scan_stem)
    session = meta.get("session", f"ses-{entities['ses']}" if "ses" in entities else "")
    task = meta.get("task", f"task-{entities['task']}" if "task" in entities else "")
    acquisition = meta.get("acquisition", f"acq-{entities['acq']}" if "acq" in entities else "")
    run = meta.get("run", f"run-{entities['run']}" if "run" in entities else "")
    scan_id = meta.get("scan_id", scan_stem)
    site_raw = meta.get("site", "")
    site = normalize_site_name(site_raw)

    return ScanRecord(
        ts_path=ts_path,
        meta_path=meta_path,
        subject=subject,
        session=session,
        task=task,
        acquisition=acquisition,
        run=run,
        site=site,
        site_raw=site_raw,
        dataset=meta.get("dataset", "unknown"),
        sex=str(meta.get("sex", "") or "").strip(),
        age=float(meta.get("age", -1) or -1),
        diag=int(diag) if diag is not None else -1,
        diag_name=diag_name,
        tr=float(meta.get("tr", 2.0) or 2.0),
        n_timepoints=int(meta.get("n_timepoints", 0)),
        scan_id=scan_id,
    )


def load_scan_records(
    data_dirs,
    diag_filter=None,
    max_scans: int | None = None,
    exclude_unknown_diagnoses: bool = False,
):
    if isinstance(data_dirs, str):
        data_dirs = [data_dirs]

    records = []
    for data_dir in data_dirs:
        ts_files = sorted(glob.glob(os.path.join(data_dir, "*_roi_ts.npy")))
        for ts_path in ts_files:
            record = _read_record(ts_path)
            if exclude_unknown_diagnoses and record.diag < 0:
                continue
            if diag_filter is not None:
                if isinstance(diag_filter, (list, tuple, set)):
                    if record.diag not in diag_filter:
                        continue
                elif record.diag != diag_filter:
                    continue
            records.append(record)

    if max_scans is not None:
        records = records[:max_scans]
    return records


def make_subject_id(record: ScanRecord):
    return f"{record.dataset}:{record.subject}"


def resample_time_window(window: np.ndarray, target_timepoints: int):
    if window.shape[0] == target_timepoints:
        return window.astype(np.float32, copy=False)
    if window.shape[0] <= 1:
        return np.repeat(window.astype(np.float32), target_timepoints, axis=0)

    old_grid = np.linspace(0.0, 1.0, num=window.shape[0], dtype=np.float32)
    new_grid = np.linspace(0.0, 1.0, num=target_timepoints, dtype=np.float32)
    resampled = np.stack(
        [np.interp(new_grid, old_grid, window[:, roi]) for roi in range(window.shape[1])],
        axis=1,
    )
    return resampled.astype(np.float32, copy=False)


def compute_sliding_windows(
    n_timepoints: int,
    tr: float,
    window_seconds: float,
    stride_seconds: float,
):
    native_window = max(4, int(round(window_seconds / max(tr, 1e-6))))
    native_window = min(native_window, n_timepoints)
    if n_timepoints <= native_window:
        return [(0, n_timepoints)]

    stride = max(1, int(round(stride_seconds / max(tr, 1e-6))))
    starts = list(range(0, n_timepoints - native_window + 1, stride))
    last_start = n_timepoints - native_window
    if starts[-1] != last_start:
        starts.append(last_start)
    return [(start, start + native_window) for start in starts]


def sample_random_window(
    ts: np.ndarray,
    tr: float,
    window_seconds: float,
):
    native_window = max(4, int(round(window_seconds / max(tr, 1e-6))))
    native_window = min(native_window, ts.shape[0])
    if ts.shape[0] <= native_window:
        return ts

    start = np.random.randint(0, ts.shape[0] - native_window + 1)
    end = start + native_window
    return ts[start:end]


class WindowedFMRIDataset(Dataset):
    def __init__(
        self,
        data_dirs,
        window_seconds: float = 90.0,
        target_timepoints: int = 45,
        sampling: str = "random",
        stride_seconds: float = 30.0,
        tokenization_mode: str = "temporal",
        diag_filter=None,
        exclude_unknown_diagnoses: bool = False,
        max_scans: int | None = None,
    ):
        assert sampling in {"random", "center", "sliding"}
        assert tokenization_mode in {"temporal", "spatial"}

        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]

        records = load_scan_records(
            data_dirs=data_dirs,
            diag_filter=diag_filter,
            exclude_unknown_diagnoses=exclude_unknown_diagnoses,
            max_scans=max_scans,
        )

        if not records:
            raise FileNotFoundError(f"No scans found in {data_dirs}")

        self.records = records
        self.window_seconds = float(window_seconds)
        self.target_timepoints = int(target_timepoints)
        self.sampling = sampling
        self.stride_seconds = float(stride_seconds)
        self.tokenization_mode = tokenization_mode

        self.window_records = None
        if self.sampling == "sliding":
            window_records = []
            for record in self.records:
                windows = compute_sliding_windows(
                    n_timepoints=record.n_timepoints,
                    tr=record.tr,
                    window_seconds=self.window_seconds,
                    stride_seconds=self.stride_seconds,
                )
                for window_index, (start, end) in enumerate(windows):
                    window_records.append(
                        WindowRecord(
                            record=record,
                            start=start,
                            end=end,
                            window_index=window_index,
                            n_windows=len(windows),
                        )
                    )
            self.window_records = window_records

    def __len__(self):
        if self.window_records is not None:
            return len(self.window_records)
        return len(self.records)

    def _native_window_len(self, record: ScanRecord):
        native = int(round(self.window_seconds / max(record.tr, 1e-6)))
        return max(4, native)

    def _extract_window(self, ts: np.ndarray, record: ScanRecord):
        native_window = min(self._native_window_len(record), ts.shape[0])
        if ts.shape[0] <= native_window:
            return ts

        if self.sampling == "center":
            start = (ts.shape[0] - native_window) // 2
            end = start + native_window
            return ts[start:end]

        return sample_random_window(
            ts=ts,
            tr=record.tr,
            window_seconds=self.window_seconds,
        )

    def __getitem__(self, idx):
        if self.window_records is None:
            record = self.records[idx]
            window_meta = None
        else:
            window_meta = self.window_records[idx]
            record = window_meta.record

        ts = np.load(record.ts_path).astype(np.float32)
        if window_meta is None:
            window = self._extract_window(ts, record)
        else:
            window = ts[window_meta.start:window_meta.end]
        window = resample_time_window(window, self.target_timepoints)

        ts_tensor = torch.from_numpy(window)
        if self.tokenization_mode == "spatial":
            ts_tensor = ts_tensor.T

        labels = {
            "scan_id": record.scan_id,
            "subject": record.subject,
            "session": record.session,
            "task": record.task,
            "acquisition": record.acquisition,
            "run": record.run,
            "site": record.site,
            "site_raw": record.site_raw,
            "dataset": record.dataset,
            "sex": record.sex,
            "age": record.age,
            "diag": record.diag,
            "diag_name": record.diag_name,
            "tr": record.tr,
            "subject_id": make_subject_id(record),
        }
        if window_meta is None:
            labels["window_index"] = 0
            labels["window_start"] = None
            labels["window_end"] = None
            labels["n_windows"] = 1
        else:
            labels["window_index"] = window_meta.window_index
            labels["window_start"] = window_meta.start
            labels["window_end"] = window_meta.end
            labels["n_windows"] = window_meta.n_windows
        return ts_tensor, labels
