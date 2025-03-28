from qtpy.QtWidgets import (
    QPushButton,
    QProgressBar,
)
from napari.utils.notifications import show_info, show_warning


class WorkerManager:
    def __init__(self, grayout_ui_list):
        self.grayout_ui_list = grayout_ui_list
        self.active_workers = []

        # Cancel button
        self.cancel_btn = QPushButton("❌ Cancel")
        self.cancel_btn.clicked.connect(self._handle_cancel)
        
        # Progress bar
        self.pbar = QProgressBar(minimum=0, maximum=1)

    @property
    def n_active(self):
        return len(self.active_workers)

    def add_active(self, worker, max_iter: int = 0):
        worker.returned.connect(self.worker_stopped)
        if hasattr(worker, "aborted"):  # Only generator workers
            worker.aborted.connect(self.worker_stopped)
        if max_iter > 0:
            worker.yielded.connect(lambda step: self.update_pbar(step))
        self.active_workers.append(worker)
        self.pbar.setMaximum(max_iter)
        self.pbar.setValue(0)
        self._grayout_ui()
        worker.start()

    def update_pbar(self, value: int):
        self.pbar.setValue(value)

    def _grayout_ui(self):
        for ui_element in self.grayout_ui_list:
            ui_element.setEnabled(False)

    def _ungrayout_ui(self):
        for ui_element in self.grayout_ui_list:
            ui_element.setEnabled(True)

    def worker_stopped(self):
        self._ungrayout_ui()
        self.pbar.setMaximum(1)
        self.active_workers.clear()

    def _handle_cancel(self):
        if len(self.active_workers) == 0:
            show_warning("Nothing to cancel.")

        for worker in self.active_workers:
            show_info("Cancelling...")
            worker.quit()

    def clear(self):
        self.active_workers.clear()
