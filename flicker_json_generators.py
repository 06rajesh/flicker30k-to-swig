import re
from os import listdir
from typing import List
import json
import csv
import random

from pathlib import Path
import nltk
from nltk.corpus import wordnet as wn
from nltk import pos_tag, word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import framenet as fn

from flickr30k_entities_utils import get_annotations, get_sentence_data

def noun2synset(noun):
    noun_syn = wn.synset_from_pos_and_offset(noun[0], int(noun[1:])).name() if re.match(r'n[0-9]*', noun) else "'{}'".format(noun)
    return noun_syn.split('.')[0]

def get_id_list_from_file(filepath:str):
    img_id_list = []
    with open(filepath) as file:
        for line in file:
            img_id_list.append(line.rstrip())
    return img_id_list

class FlickerSituJsonGenerator:
    frames_path: Path
    export_path: Path

    all_frames: dict
    all_verbs: dict

    verb_list: list

    def __init__(self, frames_path:str, export_path:str='flicker_jsons/', verbs_list_path:str = ""):
        self.frames_path = Path(frames_path)
        self.export_path = Path(export_path)

        self.all_verbs = {}
        self.all_frames = {}

        self.verb_list = []

        if verbs_list_path != '':
            with open(verbs_list_path, "r") as f:
                self.verb_list = [line[:-1] for line in f]

        self.load_all_verbs_and_frames()

    def filter_frame_elements(self, framename, elementsdict, core_only: bool = True):

        # sorted element list by count
        elementsdict = dict(sorted(elementsdict.items(), key=lambda item: item[1], reverse=True))
        elementslist = list(elementsdict.keys())

        f = fn.frame(framename)
        if core_only:
            fe = [fe.lower() for fe in f.FE.keys() if f.FE[fe].coreType == 'Core']
        else:
            fe = [key.lower() for key in f.FE.keys()]

        intersect = [value for value in elementslist if value in fe]
        # select top unique 6
        intersect = intersect[:6]

        if 'location' not in intersect and 'place' not in intersect:
            if len(intersect) >= 6:
                del intersect[-1]
            intersect.append('place')

        return intersect

    def filter_verb_frames(self, frame_counts: dict):
        sorted_dict = dict(sorted(frame_counts.items(), key=lambda item: item[1], reverse=True))
        top_2 = list(sorted_dict.keys())[:2]
        final = []
        if len(top_2) > 0:
            final.append(top_2[0])

        if len(top_2) > 1:
            if sorted_dict[top_2[1]] > sorted_dict[top_2[0]] / 3:
                final.append(top_2[1])

        return final

    def load_all_verbs_and_frames(self):
        files = listdir(self.frames_path)

        verbs = dict()
        framenet_frames = dict()
        idx = 0
        for file in files:
            filepath = self.frames_path / file
            with open(filepath) as f:
                verb_frames = json.load(f)

            for v in verb_frames.keys():
                if v not in verbs:
                    verbs[v] = dict()
                for frame in verb_frames[v]:
                    f = frame['frame']
                    keylist = [k for k in frame['elements'].keys() if len(k.split(' ')) == 1]
                    if f not in verbs[v]:
                        verbs[v][f] = 1
                    else:
                        verbs[v][f] += 1

                    # if f not in framenet_frames:
                    #     framenet_frames[f] = set(keylist)
                    # else:
                    #     framenet_frames[f].update(keylist)

                    frame_dict = {}
                    if f in framenet_frames:
                        frame_dict = framenet_frames[f]

                    keylist = set(keylist)
                    for k in keylist:
                        if k in frame_dict.keys():
                            frame_dict[k] += 1
                        else:
                            frame_dict[k] = 1

                    framenet_frames[f] = frame_dict

            idx += 1

        # filter verb frames limits to maximum 2
        for v in verbs:
            self.all_verbs[v] = self.filter_verb_frames(verbs[v])

        # filter frames FE, keeps only available in framenet definitions
        for f in framenet_frames:
            self.all_frames[f] = self.filter_frame_elements(f, framenet_frames[f], core_only=True)

    def get_first_sentence(self, text:str):
        splited = text.split('.')
        return splited[0]

    @staticmethod
    def extract_synset(phrase: str, phrase_type: List[str]):
        tagged = pos_tag(word_tokenize(phrase))
        stemmer = PorterStemmer()

        words = [pos_tuple[0] for pos_tuple in tagged if pos_tuple[1] != 'DT']

        candidates = []
        for idx in range(len(words)):
            subset = words[idx:]
            candidates.append('_'.join(subset))

        synsets = []
        candidate = ''
        for c in candidates:
            ss = wn.synsets(c, pos=wn.NOUN)
            if len(ss) > 0:
                synsets = ss
                candidate = c
                break

        if len(synsets) == 0:
            synsets = wn.synsets(phrase_type[0], pos=wn.NOUN)
            candidate = phrase_type[0]

        if len(synsets) > 0:
            selected = synsets[0]
            stemmed = stemmer.stem(candidate)
            for syn in synsets:
                if syn.name().startswith(stemmed):
                    selected = syn
                    break
            return selected
        else:
            return None

    @staticmethod
    def get_synset_id(syn):
        syn_id = 'oov'
        if syn:
            syn_id = syn.pos() + str(syn.offset())

        return syn_id

    def second_frame_orders_after_similarity_check(self, frame1:str, frame2:str):
        list1 = self.all_frames[frame1]
        list2 = self.all_frames[frame2]

        if len(list2) == 0:
            return []

        similar = 0
        uniques = []
        for role in list2:
            if role in list1:
                similar += 1
            else:
                uniques.append(role)

        similarity = similar / len(list2)
        if similarity > 0.75:
            return uniques
        else:
            return []

    def process_verbs(self):
        verb_json = {}

        if len(self.verb_list) > 0:
            verbslist = self.verb_list
        else:
            verbslist = self.all_verbs

        for idx, v in enumerate(verbslist):
            framenet = self.all_verbs[v][0]
            f = fn.frame(framenet)

            frameelements = f.FE
            order = self.all_frames[framenet][:] # copy framelist into order
            roles = {}
            secondary_frame = ''

            for o in order:
                try:
                    elems = frameelements[o.capitalize()]
                    roles[o] = {
                        'framenet': elems.name.lower(),
                        'def': self.get_first_sentence(elems.definition),
                    }
                except KeyError:
                    print(v + ', ' + framenet + ', ' + o)
                    roles[o] = {
                        'framenet': o,
                        'def': '',
                    }

            # check if secondary frame meet the requirements, if met
            # add its role to order lists
            if len(self.all_verbs[v]) > 1:
                unique_orders = self.second_frame_orders_after_similarity_check(framenet, self.all_verbs[v][1])
                if len(unique_orders) > 0:
                    f2 = fn.frame(self.all_verbs[v][1])
                    secondary_frame = self.all_verbs[v][1].lower()
                    for uo in unique_orders:
                        order.append(uo)
                        elems = f2.FE[uo.capitalize()]
                        roles[uo] = {
                            'framenet': elems.name.lower(),
                            'def': self.get_first_sentence(elems.definition),
                        }

            # set maximum number of order to 6
            if len(order) > 6:
                order = order[:6]
                print(order)

            frame_def = {
                'framenet': f.name.lower(),
                'secondary': secondary_frame,
                'def': self.get_first_sentence(f.definition),
                'order': order,
                'roles': roles,
            }

            verb_json[v] = frame_def

        return verb_json

    def process_nouns(self):
        files = listdir(self.frames_path)

        all_syns = {}
        idx = 0
        for file in files:
            filepath = self.frames_path / file
            with open(filepath) as f:
                verb_frames = json.load(f)

            for verb in verb_frames.keys():
                for frame in verb_frames[verb]:
                    elements = frame['elements']
                    framename = frame['frame']
                    allowed_roles = self.all_frames[framename]

                    for elkey in elements.keys():
                        if elkey in allowed_roles:
                            elem = elements[elkey]
                            if 'phrase' in elem:
                                syn = self.extract_synset(elem['phrase'], elem['phrase_type'])
                                if syn:
                                    gloss = [l.name() for l in syn.lemmas()]
                                    syn_id = self.get_synset_id(syn)
                                    if syn_id not in all_syns:
                                        all_syns[syn_id] = {
                                            'gloss': gloss,
                                            'def': syn.definition(),
                                        }
                            else:
                                continue

            idx += 1
        return all_syns

    def save_flicker_situ_and_classes(self):
        verbs = self.process_verbs()
        nouns = self.process_nouns()

        all = {
            'nouns': nouns,
            'verbs': verbs,
        }

        self.save_train_classes(nouns)

        flicker_situ = 'flickersitu_space'
        outfile = self.export_path / '{}.json'.format(flicker_situ)
        with open(outfile, 'w') as f:
            json.dump(all, f)

        print('Total {} verbs and {} nouns saved to file {}'.format(len(verbs), len(nouns), outfile))

    def save_train_classes(self, nouns):
        classes_file = self.export_path / 'train_classes.csv'
        with open(classes_file, 'w', encoding='UTF8') as f:
            writer = csv.writer(f)

            # write the header
            writer.writerow(['blank', 0])

            for idx, n in enumerate(nouns.keys()):
                writer.writerow([n, idx+1])

            # add oov and pad after all classes
            writer.writerow(['oov', idx+1])
            writer.writerow(['Pad', idx + 2])

        print('Finished writing {} classes to file'.format(idx))

    def save_roles_indices(self):
        if len(self.verb_list) > 0:
            verbslist = self.verb_list
        else:
            verbslist = self.all_verbs

        unique_roles = set()
        for idx, v in enumerate(verbslist):
            frames = self.all_verbs[v]
            for f in frames:
                roles = self.all_frames[f]
                unique_roles.update(roles)

        sorted_roles = sorted(list(unique_roles))

        roles_file = self.export_path / 'role_indices.txt'
        # open file in write mode
        with open(roles_file, 'w') as fp:
            for item in sorted_roles:
                # write each item on a new line
                fp.write("%s\n" % item)
            print('Finished writing {} roles to file'.format(len(unique_roles)))

class FlickerJsonCreator:
    frames_path: Path
    annotations_path: Path
    # images_path: Path
    export_path: Path
    id_list: List
    all_verbs: dict
    debug: bool

    def __init__(self, annotation_path:str,
                 frames_path:str, idlist_file:str,
                 situ_space_file:str='./flicker_jsons/flickersitu_space.json',
                 export_path:str = './flicker_jsons', debug:bool = False):
        self.frames_path = Path(frames_path)
        self.annotations_path = Path(annotation_path)
        # self.images_path = Path(images_path)
        self.export_path = Path(export_path)
        self.id_list = get_id_list_from_file(idlist_file)
        self.debug = debug

        if debug:
            random_keys = [random.choice(self.id_list) for i in range(5)]
            self.id_list = random_keys

        with open(situ_space_file) as f:
            all = json.load(f)
            self.all_verbs = all['verbs']

    # Program to find most frequent
    # element in a list
    def most_frequent(self, List):
        if len(List) == 0:
            return ''
        return max(set(List), key=List.count)

    @staticmethod
    def trim_rolename(role:str):
        if role[-1].isdigit():
            role = role[:-1]
            if role[-1] == '_':
                role = role[:-1]
        return role

    def sort_elements(self, role_orders:List, element_lists:List):

        selected_roles = set()

        for elem in element_lists:
            for role in elem:
                if role in role_orders:
                    selected_roles.add(role)

        phrase_id_counter = {r:list() for r in selected_roles}
        sorted_elements = []

        for elem in element_lists:
            roles_dict = dict.fromkeys(selected_roles, '')
            for r in selected_roles:
                try:
                    phrase = elem[r]['phrase']
                    phrase_id_counter[r].append(elem[r]['phrase_id'])

                    syn = FlickerSituJsonGenerator.extract_synset(phrase, elem[r]['phrase_type'])
                    syn_id = FlickerSituJsonGenerator.get_synset_id(syn)
                    roles_dict[r] = syn_id
                except KeyError:
                    continue
            sorted_elements.append(roles_dict)

        selected_phrase_id = {k: self.most_frequent(phrase_id_counter[k]) for k in phrase_id_counter.keys()}

        delete_keys = []
        for phid in selected_phrase_id:
            if selected_phrase_id[phid] == '':
                delete_all = True
                for el in sorted_elements:
                    if el[phid] != '':
                        delete_all = False
                        break

                if delete_all:
                    delete_keys.append(phid)
                    for el in sorted_elements:
                        del el[phid]

        if len(delete_keys) > 0:
            for k in delete_keys:
                del selected_phrase_id[k]

        return selected_phrase_id, sorted_elements

    def nutrilize_roles_with_captions(self, role_items:List, captions:List):
        sample_roles = dict.fromkeys(role_items[0].keys(), '')
        sample_caption = ''

        ideal_items = []
        relavent_captions = []

        nonideal_items = []
        nonideal_relavent_caps = []

        for idx, item in enumerate(role_items):
            ideal_item = True
            for k in item:
                if item[k] == '':
                    ideal_item = False
                elif sample_roles[k] == '':
                    sample_roles[k] = item[k]
                    sample_caption = captions[idx]

            if ideal_item:
                ideal_items.append(item)
                relavent_captions.append(captions[idx])
            else:
                nonideal_items.append(item)
                nonideal_relavent_caps.append(captions[idx])

        # add the items with empty value from the sample value
        # and add it the the ideal items
        for i in range(len(nonideal_items)):
            item_with_empty_val = nonideal_items[i]
            for k in item_with_empty_val:
                if item_with_empty_val[k] == '':
                    item_with_empty_val[k] = sample_roles[k]
            ideal_items.append(item_with_empty_val)
            relavent_captions.append(nonideal_relavent_caps[i])

        if len(ideal_items) > 3:
            return ideal_items[0:3], relavent_captions[0:3]
        else:
            for i in range(3 - len(ideal_items)):
                ideal_items.append(sample_roles)
                relavent_captions.append(sample_caption)

        return ideal_items, relavent_captions

    def merge_box(self, boxes:List):
        x1 = []
        x2 = []
        y1 = []
        y2 = []

        for box in boxes:
            x1.append(box[0])
            y1.append(box[1])
            x2.append(box[2])
            y2.append(box[3])

        return [min(x1), min(y1), max(x2), max(y2)]

    def get_bounding_box(self, boxes_dict:dict, phrase_id:str):
        no_box = [-1, -1, -1, -1]

        if phrase_id not in boxes_dict:
            return no_box

        boxes = boxes_dict[phrase_id]
        if len(boxes) == 1:
            return boxes[0]

        return self.merge_box(boxes)

    def generate_swig_json(self, idlist:List = None):
        final_json = {}

        if not idlist:
            idlist = self.id_list

        for idx, id in enumerate(idlist):
            annotation_file = self.annotations_path / '{}.xml'.format(id)
            annotation = get_annotations(annotation_file)

            # img_file = self.images_path / '{}.jpg'.format(id)
            # read image
            # self.image = cv2.imread(str(img_file))

            filepath = self.frames_path / '{}.json'.format(id)
            with open(filepath) as f:
                frames_by_verbs = json.load(f)

            for v in frames_by_verbs:
                if v in self.all_verbs:
                    sentences = frames_by_verbs[v]
                    verb_orders = self.all_verbs[v]

                    candidate_frames = [verb_orders['framenet']]
                    if verb_orders['secondary'] != '':
                        candidate_frames.append(verb_orders['secondary'])
                    candidate_elements = [s['elements'] for s in sentences if s['frame'].lower() in candidate_frames]
                    captions = [s['sentence'] for s in sentences if s['frame'].lower() in candidate_frames]

                    if len(candidate_elements) > 0:
                        bb_id, selected_elements = self.sort_elements(verb_orders['order'], candidate_elements)
                        if len(selected_elements[0].keys()) != 0:
                            nutrilized_frames, caps = self.nutrilize_roles_with_captions(selected_elements, captions)
                            bounding_boxes = {k:self.get_bounding_box(annotation['boxes'], bb_id[k]) for k in bb_id.keys()}

                            img_json = {
                                "bb": bounding_boxes,
                                "caption": max(caps, key=len),
                                "captions": caps,
                                "height": annotation['height'],
                                "width": annotation['width'],
                                "verb": v,
                                "frames": nutrilized_frames,
                            }

                            final_json[v + '_' + id] = img_json

            # if idx > 5:
            #     break

        return final_json

    def generate_and_save_json(self, filename:str):
        all_json = self.generate_swig_json()

        if filename == 'val':
            filename = 'dev'

        outfile = self.export_path / '{}.json'.format(filename)
        with open(outfile, 'w') as f:
            json.dump(all_json, f)

        print('Total {} jsons saved to file {}'.format(len(all_json.keys()), outfile))
