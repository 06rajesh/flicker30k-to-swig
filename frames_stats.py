import json
from os import listdir
from pathlib import Path

from flickr30k_entities_utils import get_annotations, get_sentence_data

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
            frames = json.load(f)

            for item in frames:
                if item['verb'] in verbs:
                    verbs[item['verb']] += 1
                else:
                    verbs[item['verb']] = 1

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


def debug_img(imgid):
    sentence_file = 'annotations/Sentences/{}.txt'.format(imgid)
    sentence = get_sentence_data(sentence_file)
    print(len(sentence))

    frame_file = 'annotations/Frames/{}.json'.format(imgid)
    with open(frame_file) as f:
        frames = json.load(f)

    verb_sen = {}
    for s in frames:
        for f in s['frames']:
            if f['verb'] in verb_sen:
                sens = verb_sen[f['verb']]
                sens.append(s['sentence'])
            else:
                verb_sen[f['verb']] = [s['sentence']]
    print(verb_sen)

if __name__ == '__main__':
    # frames_root = Path('annotations/Frames')
    # export_to = Path('stats')
    # extract_verb_stats(frames_root, export_to)
    # extract_role_stats(frames_root, export_to)
    debug_img('8136850076')

