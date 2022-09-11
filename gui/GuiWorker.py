import os
import signal
import time
import tkinter as tk

from multiprocessing import Queue, Event
from threading import Thread

from tkinter import ttk

from gui.GuiStyling import default
from gui.GuiUtils import print_message
from process.ProcessModel import ProcessModel


class GuiWorker:
    """
    Class for representing the Progress of the Process Worker
    """

    def __init__(self, name="My Worker", width=760, height=480,
                 styling=default(), keep_open=True, keep_open_time=50):
        """
        Initialize a Gui Worker to attach to the Process Worker
        :param name: Title of the TK Window
        :param width: Width of the TK Window
        :param height: Height of the TK Window
        :param styling: Styling for the GUI Worker
        :param keep_open: If Process Worker finishes
                        close the TK Window or not
        :param keep_open_time: If keep_open = False,
                        the time the window stays
                        open when processes finish
        """

        self.thread_count = 1
        self.loop_count = 1
        self.max_progress = 100
        self.name = name
        self.width = width
        self.height = height
        self.progress = []
        self.keep_open = keep_open
        self.keep_open_time = keep_open_time
        self.styling = styling
        self.root = None
        self.main_event = None
        self.progress_pid = 0
        self.progress_worker = None
        self.incoming_data = Queue()
        self.process_manager = None

    def start_worker(self, process_manager, thread_count, loop_count, max_progress, event):
        """
        Start the Gui Worker Process with given arguments
        :param process_manager: Process Manager of the Process Worker
        :param thread_count: Thread Count of the Process Worker
        :param loop_count: Loop Count of the Process Worker
        :param max_progress: Maximum Progress of the Process Worker
        :param event: Multiprocessing Event to stop processes if needed
        """

        self.process_manager = process_manager
        self.thread_count = thread_count
        self.loop_count = loop_count
        self.max_progress = max_progress
        self.main_event = event

        progress_worker = Thread(target=self.create_window)
        progress_worker.start()

    def terminate_worker(self, forced=True):
        """
        If the TK Window gets closed, terminate all workers with event.set()
        """

        print_message("Terminating all Workers...")

        if forced:
            os.kill(self.progress_pid, signal.SIGTERM)

            self.main_event.set()
            self.root.destroy()

            quit(0)
        elif not self.keep_open:
            time.sleep(self.keep_open_time)
            self.root.destroy()

    def create_window(self):
        """
        Create a TK Inter Window with given parameters
        """

        self.root = tk.Tk()
        self.root.geometry(str(self.width) + "x" + str(self.height))
        self.root.title(self.name)

        self.root.protocol("WM_DELETE_WINDOW", self.terminate_worker)

        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=3)

        self.progress = self.create_progress_bars(self.root)

        progress_worker = Thread(target=self.do_progress, args=(self.update_progress,))
        progress_worker.start()

        self.progress_worker = progress_worker

        try:
            self.root.mainloop()
        except Exception:
            print_message("Shutting down TkInter")

    def create_progress_bars(self, root):
        """
        Create one pair of UI Elements for each process
        :param root: The TKINTER Root
        :return: An Array of Tuples (prefix: Label, Progressbar, suffix: Label)
        """

        progress = []

        for thread in range(self.thread_count):
            progress_bar = ttk.Progressbar(
                root,
                orient='horizontal',
                value=0,
                length=400,
            )

            process_label = ttk.Label(root, text="")
            process_info_label = ttk.Label(root, text="")

            process_label.grid(column=0, row=thread, sticky=tk.W, padx=5, pady=5)
            progress_bar.grid(column=1, row=thread, sticky=tk.W, padx=10, pady=10)
            process_info_label.grid(column=2, row=thread, sticky=tk.W, padx=5, pady=5)

            progress.append((process_label, progress_bar, process_info_label))

        return progress

    def get_prefix(self, process_model: dict):
        """
        Get the customized prefix from the Gui Styling (Object)
        :param process_model: The process model converted to a dict
        :return: Prefix (String)
        """

        return self.styling.get_prefix(process_model)

    def get_suffix(self, process_model: dict):
        """
        Get the customized suffix from the Gui Styling (Object)
        :param process_model: The process model converted to a dict
        :return: Suffix (String)
        """

        return self.styling.get_suffix(process_model)

    def get_progress(self, process_model: ProcessModel):
        """
        Calculate the progress of a certain
        :param process_model: The model of the process
        :return: a percentage from (0-100%)
        """

        return (process_model.progress / float(self.max_progress)) * 10

    def update_progress(self, process: ProcessModel):
        """
        Updates the progress of all worker processes
        """

        if self.progress:
            ui_elements = self.progress[process.pid]

            ui_elements[0]['text'] = self.get_prefix(process.__dict__)
            ui_elements[2]['text'] = self.get_suffix(process.__dict__)
            ui_elements[1]['value'] = self.get_progress(process)

    def do_progress(self, callback):
        """
        Worker for updating the progress
        :param callback: Callback for updating progress
        """

        self.progress_pid = os.getpid()

        while True:
            while self.incoming_data.qsize() > 0:
                callback(self.incoming_data.get(timeout=1))

            if self.main_event.is_set():
                self.terminate_worker(forced=False)
                break

