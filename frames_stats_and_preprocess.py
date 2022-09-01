import json
from os import listdir
from pathlib import Path

from flickr30k_entities_utils import get_annotations, get_sentence_data
from flicker_frames_preprocessing import FlickerFramesPreprocessor, FramesPreprocessingPathArguments
from flicker_to_swig_utils import FlickerSentenceSwigFramer

def sort_dict_by_val(x:dict):
    return {k: v for k, v in sorted(x.items(), key=lambda item: item[1], reverse=True)}

def save_ouput_to_json(output, outfile):
    with open(outfile, 'w') as f:
        json.dump(output, f, indent=2)

def extract_verb_stats(rootpath:Path, exportpath: Path):
    files = listdir(rootpath)

    verbs = {}
    for file in files:
        filepath = rootpath / file
        with open(filepath) as f:
            frames_by_verbs = json.load(f)

            for verb in frames_by_verbs:
                if verb in verbs:
                    verbs[verb] += 1
                else:
                    verbs[verb]  = 1

    verbs = sort_dict_by_val(verbs)

    export_file = exportpath / "verb_indices.json"
    save_ouput_to_json(verbs, export_file)

def extract_role_stats(rootpath:Path, exportpath: Path):
    files = listdir(rootpath)

    roles = {}
    idx = 0
    for file in files:
        filepath = rootpath / file
        with open(filepath) as f:
            frames = json.load(f)
            for item in frames:
                elems = item['elements']
                for key in elems:
                    if key in roles:
                        roles[key] += 1
                    else:
                        roles[key] = 1
        idx += 1
        # if idx >= 5:
        #     break

    roles = sort_dict_by_val(roles)

    export_file = exportpath / "role_indices.json"
    save_ouput_to_json(roles, export_file)

def preprocess_frames(targetfile:str, export_dir: str = 'annotations/Frames_processed'):
    img_id_list = []
    with open(targetfile) as file:
        for line in file:
            img_id_list.append(line.rstrip())

    paths = FramesPreprocessingPathArguments(
        annot_root='annotations',
        frames_dir='annotations/Frames',
        export_dir=export_dir,
        imgs_dir='flickr30k-images'
    )

    for id in img_id_list:
        processor = FlickerFramesPreprocessor(id, paths)
        processor.process_and_save_frames()


if __name__ == '__main__':

    # sentence_file = 'annotations/Sentences/{}.txt'.format('142786312')
    # sentence = get_sentence_data(sentence_file)
    #
    # sentences = []
    # for item in sentence:
    #     sentences.append(item['sentence'])
    #
    # framer = FlickerSentenceSwigFramer('annotations/Sentences', 'annotations/Frames')
    #
    # for s in sentences:
    #     print(s)
    #     verb_idx, verbs = framer.get_verb_idx(s, with_frames=True)
    #     print("=================================")

    # sample = 'a guy sitting in a chair with a mug on the table next to him'
    # verb_idx, verbs = framer.get_verb_idx(sample, with_frames=True)
    # frames = framer.detect_frames_with_custom_locs(sample, verb_idx)
    # print(frames)

    # vbg = framer.get_swig_frames_from_verb('sideway')
    # print(vbg)

    # preprocess_frames('idlists/val.txt', 'annotations/Frames_processed')
    # preprocess_frames('idlists/test.txt', 'annotations/Frames_processed')
    # preprocess_frames('idlists/train.txt', 'annotations/Frames_processed')

    frames_root = Path('annotations/Frames_processed')
    export_to = Path('stats')
    extract_verb_stats(frames_root, export_to)
    # extract_role_stats(frames_root, export_to)
