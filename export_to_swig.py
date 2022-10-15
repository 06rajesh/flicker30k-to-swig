import re
import os
from os import listdir
from typing import List
import json
import cv2
import random

from pathlib import Path
from nltk.corpus import wordnet as wn

from flickr30k_entities_utils import get_annotations, get_sentence_data
from flicker_json_generators import FlickerSituJsonGenerator, FlickerJsonCreator

def noun2synset(noun):
    noun_syn = wn.synset_from_pos_and_offset(noun[0], int(noun[1:])).name() if re.match(r'n[0-9]*', noun) else "'{}'".format(noun)
    return noun_syn.split('.')[0]


def generate_verb_indices(verbcount:str, minumum_count:int=75):
    verbcount_file = Path(verbcount)
    verbs = []
    with open(verbcount_file) as f:
        all_verbs = json.load(f)

    exclude = ['striping', 'sleeving', 'multicoloring']

    for v in all_verbs:
        if all_verbs[v] > minumum_count and v not in exclude:
            verbs.append(v)

    # open file in write mode
    with open('./flicker_jsons/verb_indices.txt', 'w') as fp:
        for item in verbs:
            # write each item on a new line
            fp.write("%s\n" % item)
        print('Finished writing {} verbs to file'.format(len(verbs)))


def debug_json_file(filename:str, img_dir:str):
    with open(filename) as f:
        all = json.load(f)

    with open('flicker_jsons/flickersitu_space.json') as f2:
        flicker_json = json.load(f2)

    all_verbs = flicker_json['verbs']

    img_root = Path(img_dir)

    key_lists = list(all.keys())
    random_keys = [random.choice(key_lists) for i in range(5)]

    for key in random_keys:
        img_id = key.split('_')[1]
        verb = key.split('_')[0]
        img_path = img_root / '{}.jpg'.format(img_id)

        # print(all[key])

        # frames_path = Path('annotations/Frames_processed')
        # filepath = frames_path / '{}.json'.format(img_id)
        # with open(filepath) as f:
        #     verb_frames = json.load(f)
        #
        # print(verb_frames[verb])
        #

        annotation = all[key]
        verb = annotation['verb']

        print(key)
        print(annotation['caption'])
        print(verb)
        print(annotation['bb'].keys())
        print(all_verbs[verb]['order'])

        frame = annotation['frames'][0]
        for role in frame:
            noun = noun2synset(frame[role])
            print(role + ': ' + noun)

        # read image
        img = cv2.imread(str(img_path))

        result = img.copy()
        boxes = annotation['bb']
        for key in boxes:
            b = boxes[key]
            cv2.rectangle(result, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)

        # show thresh and result
        cv2.imshow("bounding_box", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def fill_empty_orders(filename:str, situ_space:str='flicker_jsons/flickersitu_space.json'):
    with open(filename) as f:
        all = json.load(f)

    with open(situ_space) as f2:
        flicker_json = json.load(f2)

    all_verbs = flicker_json['verbs']

    key_lists = list(all.keys())
    updated = 0
    for key in key_lists:
        annotation = all[key]
        verb = annotation['verb']
        verb_orders = all_verbs[verb]['order']
        bb_keys = annotation['bb'].keys()
        if len(bb_keys) != len(verb_orders):
            excluded = []
            for o in verb_orders:
                if o not in bb_keys:
                    excluded.append(o)
            temp_bb = annotation['bb']
            temp_frames = annotation['frames']

            for ex in excluded:
                temp_bb[ex] = [-1, -1, -1, 1]
                for frame in temp_frames:
                    frame[ex] = ''

            annotation['bb'] = temp_bb
            annotation['frames'] = temp_frames

            updated += 1
            all[key] = annotation

    if updated > 0:
        with open(filename, 'w') as f:
            json.dump(all, f)

        print('Total {} entries updated and saved to file {}'.format(updated, filename))
    else:
        print('Total {} entries updated.'.format(updated))



def get_max_len_caption(captions:List[str], fallback=''):
    if not captions:
        return fallback
    max_str = captions[0]
    for x in captions:
        if len(x) > len(max_str):
            max_str = x
    return max_str

def check_max_len(dataset, target_dir="flicker_jsons/"):
    dataset_to_file = {
        "validation": "dev.json",
        "test": "test.json",
        "train": "train.json"
    }

    target = dataset_to_file[dataset]
    generated = Path(target_dir)

    generated_target = generated / target
    if not os.path.exists(generated_target):
        raise ValueError("Generated target file does not exists")

    generated_file = open(generated_target)
    generated_json = json.load(generated_file)

    max_len = 0
    for key in generated_json.keys():
        captions = generated_json[key]["captions"]
        max_len_caption = get_max_len_caption(captions)
        caption_len = len(max_len_caption.split())
        if caption_len > max_len:
            max_len = caption_len
    generated_file.close()

    print("=============================")
    print(max_len)
    print("=============================")

if __name__ == '__main__':
    # check_max_len("validation")
    # with open('./flicker_jsons/flickersitu_space.json') as f:
    #     all = json.load(f)
    #     nouns = all['nouns']
    #     verb_orders = all['verbs']

    # place_count = 0

    # for v in verb_orders:
    #     print(verb_orders[v])
    #     break
        # if 'place' in verb_orders[v]['order']:
        #     place_count += 1
    #
    # print(place_count, len(verb_orders))
    #
    # sample_verbs = {}
    # for idx, key in enumerate(verb_orders.keys()):
    #     start = 0
    #     if idx >= start:
    #         sample_verbs[key] = verb_orders[key]
    #         if idx > start+5:
    #             break
    # print(sample_verbs)
    #
    # sample_nouns = {}
    # for idx, key in enumerate(nouns.keys()):
    #     start = 70
    #     if idx >= start:
    #         sample_nouns[key] = nouns[key]
    #         if idx > start+5:
    #             break
    # print(sample_nouns)

    # generate_verb_indices('stats/verb_indices.json')

    # generator = FlickerSituJsonGenerator('annotations/Frames_processed',
    #                                      verbs_list_path='flicker_jsons/verb_indices.txt')
    # generator.save_flicker_situ_and_classes()
    # generator.save_roles_indices()


    """
    dev.json: 3139
    test.json: 3095
    train.json: 92906
    """
    targetfile_without_ext = ''

    if targetfile_without_ext == 'val' or targetfile_without_ext == 'test' or targetfile_without_ext == 'train':
        creator = FlickerJsonCreator(
            annotation_path='annotations/Annotations',
            frames_path='annotations/Frames_processed',
            idlist_file='idlists/{}.txt'.format(targetfile_without_ext),
            debug=False
        )

        creator.generate_and_save_json(targetfile_without_ext)
        # all_json = creator.generate_swig_json()
        # print(all_json)

    # fill_empty_orders('SWiG_jsons/dev.json', 'SWiG_jsons/imsitu_space.json')
    # fill_empty_orders('flicker_jsons/test.json')
    debug_json_file('flicker_jsons/train.json', 'flickr30k-images/')

    # img_id = '6229825733'
    # frames_path = Path('annotations/Frames')
    # filepath = frames_path / '{}.json'.format(img_id)
    # with open(filepath) as f:
    #     verb_frames = json.load(f)
    #
    # print(verb_frames)

