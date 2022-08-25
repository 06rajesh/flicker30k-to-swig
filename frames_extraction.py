import multiprocessing
import time
import json

from pathlib import Path
from tqdm import tqdm
from flicker_to_swig_utils import FlickerSentenceSwigFramer

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def save_log_to_json(logfile, logval):
    with open(logfile, 'w') as f:
        json.dump(logval, f)

def get_id_list_from_file(filepath:str):
    img_id_list = []
    with open(filepath) as file:
        for line in file:
            img_id_list.append(line.rstrip())
    return img_id_list

def export_flicker_to_frame_by_id(idlist: list[str], start_from=0, sentences_dir:str='annotations/Sentences', export_dir='annotations', batch_size=4, target_file:str= ''):
    export_path = Path(export_dir)
    export_folder = export_path / 'Frames'
    logfile = export_path / 'log.txt'

    flickerToSwigger = FlickerSentenceSwigFramer(
        sentences_dir=sentences_dir,
        export_dir=str(export_folder),
    )

    processed_batches = (start_from) // batch_size

    log = {
        'target': target_file,
        'batch_size': batch_size,
        'last_processed_batch': processed_batches,
        'last_saved_index': 0,
        'last_saved_id': '',
    }


    batches = list(chunks(idlist[start_from:], batch_size))

    idx = 0
    for batch in tqdm(batches):
        # create processes
        processes = [multiprocessing.Process(target=flickerToSwigger.export_sentence_frames(imgid), args=[imgid]) for imgid in batch]

        # start the processes
        for process in processes:
            process.start()

        # wait for completion
        for process in processes:
            process.join()

        log['last_processed_batch'] = processed_batches+idx
        log['last_saved_index'] = start_from + (idx+1)*batch_size - 1
        log['last_saved_id'] = batch[-1]

        save_log_to_json(logfile, log)
        idx += 1

if __name__ == '__main__':

    filename = 'idlists/val.txt'
    img_id_list = get_id_list_from_file(filename)

    export_flicker_to_frame_by_id(
        img_id_list,
        start_from=988,
        sentences_dir='annotations/Sentences',
        export_dir='annotations',
        batch_size=4,
        target_file=filename
    )
