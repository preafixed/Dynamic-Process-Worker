import json
from multiprocessing.synchronize import Event

import spacy
import random

from multiprocessing import Queue
from spacy.util import minibatch, compounding
from spacy.training import Example

from gui.GuiStyling import GuiStyling
from gui.GuiUtils import calculate_time
from gui.GuiWorker import GuiWorker
from process.ProcessManager import ProcessManager
from process.ProcessWorker import ProcessWorker

spacy.require_gpu()

nlp = spacy.load("en_core_web_sm")
ner = nlp.get_pipe("ner")

path = "C:\\Users\\"
trainingDataPath = path + "\\" + "trainingData.json"


def do_work(thread_id, loop_count, event: Event, queue: Queue, process_manager: ProcessManager, args):
    def close_process():
        """
        If needed, push data to the result queue
        """

        process = process_manager.get_process(thread_id)
        calculated_time = calculate_time(process.time_started)

        queue.put(
            {
                "process_id": thread_id,
                "process_result": calculated_time
            }
        )

        save_process(nlp)

    for i in range(loop_count):
        training_data = args[0]

        random.shuffle(training_data)
        losses = {}

        loss_value = 0
        loss_count = 0

        batches = minibatch(training_data, size=compounding(4.0, 32.0, 1.001))
        batch_iteration = 0

        for batch in batches:
            batch_iteration += 1

            for text, annotations in batch:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, {"entities": annotations.get("entities")})
                nlp.update([example], losses=losses, drop=0.3)

                loss_value += round(losses.get("ner"))
                loss_count += 1

                losses = {
                    "loss_value": loss_value,
                    "loss_count": loss_count
                }

                process_manager.update(thread_id, progress=batch_iteration, iteration=i, args=losses)

                if event.is_set():
                    close_process()
                    break

    close_process()


def save_process(result):
    nlp.to_disk("pipeline")
    print(result)


def load_annotations(training_data):
    for _, annotations in training_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])


def start_process_worker_gui():
    """
    Start the Process Worker with a GUI and call retrieve_results() when
    the processes have all finished
    """

    with open(trainingDataPath, 'r', encoding='utf8') as file:

        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
        with nlp.disable_pipes(*other_pipes):

            training_data = json.load(file)
            load_annotations(training_data)

            gui_styling = GuiStyling("Process [ID:%PID] (%PROGRESS%)",
                                     "Iteration [%ITER/%MAX_ITER], Time [%TIME], Ticks [%TICKS],"
                                     "Losses [%LOSSES]", [("%LOSSES", "loss_value")])

            gui_worker = GuiWorker(styling=gui_styling)

            ProcessWorker(do_work, training_data, gui_worker=gui_worker, max_progress=40,
                          thread_count=1, loop_count=10)


if __name__ == '__main__':
    start_process_worker_gui()
