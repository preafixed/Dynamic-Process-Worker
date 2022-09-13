import json
from multiprocessing.synchronize import Event

import spacy
import random

from multiprocessing import Queue

from spacy.scorer import Scorer
from spacy.util import minibatch, compounding
from spacy.training import Example

from src.gui.GuiStyling import GuiStyling
from src.gui.GuiUtils import calculate_time
from src.gui.GuiWorker import GuiWorker
from src.process.ProcessManager import ProcessManager
from src.process.ProcessWorker import ProcessWorker

# spacy.require_gpu()

nlp = spacy.load("en_core_web_sm")
ner = nlp.get_pipe("ner")

path = "/trainingIn"
trainingDataPath = path + "\\" + "trainingData.json"


def do_work(thread_id, loop_count, event: Event, queue: Queue, process_manager: ProcessManager, args):
    def close_process(_args):
        process = process_manager.get_process(thread_id)
        calculated_time = calculate_time(process.time_started)
        scorer = Scorer(nlp)

        queue.put(
            {
                "process_id": thread_id,
                "process_time": str(calculated_time),
            }
        )

        save_process(nlp, scorer.score(_args))

    training_data = args[0]
    examples = []

    for i in range(loop_count):

        losses = {}
        examples = []
        loss_value = 0
        loss_count = 0
        training_progress = 0

        random.shuffle(training_data)
        batches = minibatch(training_data, size=compounding(4., 3800., 1.001))

        for batch in batches:
            for text, annotations in batch:
                doc = nlp.make_doc(text)
                annotations: dict = annotations.get("entities")

                example = Example.from_dict(doc, {"entities": annotations})

                nlp.update([example], losses=losses, drop=0.3)

                example.predicted = nlp(str(example.predicted))

                if losses:
                    loss_value += round(losses.get("ner"))
                    loss_count += 1

                progress = (training_progress / (len(training_data) - 1)) * 100

                losses = {
                    "loss_value": loss_value,
                    "loss_count": loss_count
                }

                process_manager.update(thread_id, progress=progress, iteration=i, args=losses)

                examples.append(example)

                if event.is_set():
                    close_process(examples)
                    break

                training_progress += 1

    close_process(examples)


def extract_score(result_dict):
    return {
        "ents_p": result_dict["ents_p"],
        "ents_r": result_dict["ents_r"],
        "ents_f": result_dict["ents_f"]
    }


def save_process(_nlp, score):
    _nlp.to_disk("pipeline")

    extracted_result = extract_score(score)
    pretty_result = json.dumps(extracted_result, indent=4)

    print(pretty_result)


def process_finished(results):
    for result in results:
        pretty_result = json.dumps(result, indent=4)

        print(pretty_result)


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
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner" or pipe != "tok2vec"]
        with nlp.disable_pipes(*other_pipes):
            training_data = json.load(file)
            load_annotations(training_data)

            queue = Queue()

            gui_styling = GuiStyling("Process [ID:%PID] (%PROGRESS%)",
                                     "Iteration [%ITER/%MAX_ITER], Time [%TIME], Ticks [%TICKS],"
                                     "Losses [%LOSSES]", [("%LOSSES", "loss_value")])

            gui_worker = GuiWorker(name="Spacy NLP Worker", styling=gui_styling, width=860, height=480)

            ProcessWorker(do_work, training_data, queue, gui_worker=gui_worker,
                          thread_count=1, loop_count=1, callback=process_finished)


if __name__ == '__main__':
    start_process_worker_gui()
