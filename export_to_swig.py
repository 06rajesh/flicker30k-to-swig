import re
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


def generate_verb_indices(verbcount:str, minumum_count:int=70):
    verbcount_file = Path(verbcount)
    verbs = []
    with open(verbcount_file) as f:
        all_verbs = json.load(f)

    for v in all_verbs:
        if all_verbs[v] > minumum_count:
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

    img_root = Path(img_dir)

    key_lists = list(all.keys())
    random_keys = [random.choice(key_lists) for i in range(5)]

    for key in random_keys:
        img_id = key.split('_')[1]
        img_path = img_root / '{}.jpg'.format(img_id)

        annotation = all[key]

        print(annotation['caption'])
        print(annotation['verb'])

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

if __name__ == '__main__':
    # with open('./flicker_jsons/flickersite_space.json') as f:
    #     all = json.load(f)
    #     nouns = all['nouns']
    #     verb_orders = all['verbs']
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
    #
    # generator = FlickerSituJsonGenerator('annotations/Frames_processed',
    #                                      verbs_list_path='flicker_jsons/verb_indices.txt')
    # generator.save_flicker_situ_and_classes()
    # generator.save_roles_indices()


    """
    dev.json: 3480
    test.json: 3394
    train.json: 102048
    """
    targetfile_without_ext = ''

    if targetfile_without_ext == 'val' or targetfile_without_ext == 'test' or targetfile_without_ext == 'train':
        creator = FlickerJsonCreator(
            annotation_path='annotations/Annotations',
            frames_path='annotations/Frames_processed',
            idlist_file='idlists/{}.txt'.format(targetfile_without_ext),
        )

        creator.generate_and_save_json(targetfile_without_ext)

    debug_json_file('flicker_jsons/train.json', 'flickr30k-images/')

