from typing import List
import os
import json

import cv2
from pathlib import Path
from flickr30k_entities_utils import get_annotations, get_sentence_data
import spacy

class FramesPreprocessingPathArguments:
    annotations: Path
    frames: Path
    export: Path
    sentences: Path
    images: Path

    def __init__(self, annot_root:str, frames_dir:str = 'annotations/Frames', export_dir:str = 'annotations/Frames_preprocessed', imgs_dir:str = 'flicker30k-images'):
        self.frames = Path(frames_dir)
        self.export = Path(export_dir)
        self.images = Path(imgs_dir)

        annot_root = Path(annot_root)
        self.annotations = annot_root / 'Annotations'
        self.sentences = annot_root / 'Sentences'

        if not os.path.exists(self.export):
            os.mkdir(self.export)


class FlickerFramesPreprocessor:
    paths: FramesPreprocessingPathArguments
    imgid: str
    debug: bool

    sentences: dict
    annotations: dict
    image: any
    frames: dict

    def __init__(self, imgid:str, pathargs: FramesPreprocessingPathArguments, debug:bool=False):
        self.paths = pathargs
        self.imgid = imgid
        self.debug = debug

        self.nlp = spacy.load("en_core_web_sm")

        self.load_img_properties()

    def load_img_properties(self):
        sentence_file = self.paths.sentences / '{}.txt'.format(self.imgid)
        self.sentences = get_sentence_data(sentence_file)

        annotation_file = self.paths.annotations / '{}.xml'.format(self.imgid)
        self.annotations = get_annotations(annotation_file)

        img_file =  self.paths.images / '{}.jpg'.format(self.imgid)
        # read image
        self.image = cv2.imread(str(img_file))

        frame_file = self.paths.frames / '{}.json'.format(self.imgid)
        with open(frame_file) as f:
            try:
                self.frames = json.load(f)
            except json.decoder.JSONDecodeError:
                print(self.imgid)
                print(frame_file)
                exit()

    def _save_ouput_to_json(self, output):
        outfile = self.paths.export / '{}.json'.format(self.imgid)
        with open(outfile, 'w') as f:
            json.dump(output, f)

    def bb_list_to_dict(self, bblist:List):
        """
        bblist : List
            items: ['x1', 'y1', 'x2', 'y2']
            The (x1, y1) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner
        """
        return {
            'x1': bblist[0], 'x2': bblist[2], 'y1': bblist[1], 'y2': bblist[3],
        }

    def get_area_covers(self, bblist1:List, bblist2:List):
        """
        Calculate the how much area of bounding box two covers inside bounding box 1.

        Parameters
        ----------
        bblist1 : List
            items: ['x1', 'y1', 'x2', 'y2']
            The (x1, y1) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner
        bblist2 : List
            items: ['x1', 'y1', 'x2', 'y2']
            The (x, y) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner

        Returns
        -------
        float
            in [0, 1]
        """
        bb1 = self.bb_list_to_dict(bblist1)
        bb2 = self.bb_list_to_dict(bblist2)

        assert bb1['x1'] < bb1['x2']
        assert bb1['y1'] < bb1['y2']
        assert bb2['x1'] < bb2['x2']
        assert bb2['y1'] < bb2['y2']

        # determine the coordinates of the intersection rectangle
        x_left = max(bb1['x1'], bb2['x1'])
        y_top = max(bb1['y1'], bb2['y1'])
        x_right = min(bb1['x2'], bb2['x2'])
        y_bottom = min(bb1['y2'], bb2['y2'])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        # The intersection of two axis-aligned bounding boxes is always an
        # axis-aligned bounding box
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # compute the area of both AABBs
        bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
        bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = intersection_area / float(bb1_area)
        assert iou >= 0.0
        assert iou <= 1.0
        return iou

    def get_box_area(self, box: List):
        """
        bblist : List
           items: ['x1', 'y1', 'x2', 'y2']
           The (x1, y1) position is at the top left corner,
           the (x2, y2) position is at the bottom right corner
        """
        return (box[2] - box[0]) * (box[3] - box[1])

    def extract_phrases(self, sentence: str, phrases_list:List):
        start = 0
        selected = []
        for item in phrases_list:
            spart = sentence[start:]
            phrase = item['phrase']
            phrase_pos = spart.find(phrase)
            if phrase_pos != -1:
                selected.append(item)

        if len(selected) == 0:
            for item in phrases_list:
                phrase = item['phrase']
                s_pos = phrase.find(sentence)
                if s_pos != -1:
                    selected.append(item)
        return selected

    def merge_phrase_by_box(self, phrases: list):

        boxes = self.annotations['boxes']

        # Separate the boxes from their phrase group
        # boxes is a list of dicts, some phrases might
        # contain more than one box
        phrase_box_count = {}
        all_boxes = {}
        for item in phrases:
            phrid = item['phrase_id']
            # skip notvisual phrase
            if phrid == '0' or phrid not in boxes:
                continue
            item_boxes = boxes[phrid]
            phrase_box_count[phrid] = len(item_boxes)
            for i, b in enumerate(item_boxes):
                box_key = phrid + "_" + str(i)
                all_boxes[box_key] = {
                    'phrase_id': phrid,
                    'box': b,
                    'area': self.get_box_area(b),
                }

        merged_boxes = []
        for bkey in all_boxes.keys():
            bitem = all_boxes[bkey]
            found_parent = False
            for scnkey in all_boxes.keys():
                if scnkey == bkey:
                    continue
                seconditem = all_boxes[scnkey]
                iou = self.get_area_covers(bitem['box'], seconditem['box'])
                if iou > 0.85:
                    found_parent = True
                    biu = self.get_area_covers(seconditem['box'], bitem['box'])
                    if biu > 0.85 and iou >= biu:
                        merged_boxes.append(bitem)
                    break

            if not found_parent:
                merged_boxes.append(bitem)

        removed_duplicates = []
        for bitem in merged_boxes:
            has_duplicate = False
            current_id = bitem['phrase_id']
            for sitem in merged_boxes:
                second_id = sitem['phrase_id']
                if current_id == second_id:
                    continue
                if bitem['box'] == sitem['box']:
                    if phrase_box_count[current_id] < phrase_box_count[second_id]:
                        has_duplicate = True
                    break
            if not has_duplicate:
                removed_duplicates.append(bitem)

        selected = ''
        max_area = 0
        for boxitem in removed_duplicates:
            if selected == boxitem['phrase_id']:
                continue
            if boxitem['area'] > max_area:
                selected = boxitem['phrase_id']

        selected_phrase = {}
        for item in phrases:
            if item['phrase_id'] == selected:
                selected_phrase = item
                break
        return selected_phrase

    def get_root_from_complex_phrase(self, phrase:str):
        comma_splitted = phrase.split(',')

        phrases = []
        for splitter in comma_splitted:
            added = False
            splitted = splitter.split(' and ')
            if len(splitted) > 1:
                added = True
                phrases.extend(splitted)

            splitted = splitter.split(' or ')
            if len(splitted) > 1:
                added = True
                phrases.extend(splitted)

            splitted = splitter.split(' but ')
            if len(splitted) > 1:
                added = True
                phrases.extend(splitted)

            if not added:
                phrases.append(splitter.strip())

        for ph in phrases:
            doc = self.nlp(ph)
            selected = None
            for token in doc:
                if token.dep_ == 'ROOT':
                    selected = token.text
                    break

            if selected:
                return selected

        return None

    def detect_phrase(self, phrase:str, phrases:List):
        root = self.get_root_from_complex_phrase(phrase)
        if root:
            for p in phrases:
                if root in p['phrase']:
                    return p
        return self.merge_phrase_by_box(phrases)

    def select_single_phrase(self, phrase:str, phrases:List):
        phrase_list = self.extract_phrases(phrase, phrases)
        if len(phrase_list) > 1:
            selected = self.detect_phrase(phrase, phrase_list)
        elif len(phrase_list) > 0:
            selected = phrase_list[0]
        else:
            selected = {}
        return selected

    def sort_frames_by_verbs(self, frames: List):
        frames_by_verbs = {}
        for item in frames:
            item_frames = item['frames']
            for f in item_frames:
                verb = f['verb']
                edited = {
                    'sentence': item['sentence'],
                    'frame': f['frame'],
                    'elements': f['elements']
                }

                modified_frames = []
                if verb in frames_by_verbs:
                    modified_frames = frames_by_verbs[verb]

                modified_frames.append(edited)
                frames_by_verbs[verb] = modified_frames

        return frames_by_verbs

    def view_frame_on_image(self, frame: dict):
        result = self.image.copy()
        boxes = self.annotations['boxes']

        phrase_ids = []
        for fkey in frame:
            try:
                phid = frame[fkey]['phrase_id']
            except KeyError:
                print(fkey + ': "" ')
                continue
            phrase_ids.append(phid)
            print(fkey + ':' + frame[fkey]['phrase'])

        for key in phrase_ids:
            try:
                for b in boxes[key]:
                    cv2.rectangle(result, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            except KeyError:
                print('Box with phrase id {} not found. Can not able to display.'.format(key))


        # show thresh and result
        cv2.imshow("bounding_box", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def generalize_keys(self, key:str):
        if 'location' in key:
            key = 'location'
        return key

    def get_process_frames(self):
        processed_entries = []
        for s_idx, s in enumerate(self.frames):
            current = self.sentences[s_idx]
            phrases = current['phrases']
            modifiled_frames = []
            for item in s['frames']:
                mod = {
                    'frame': item['frame'],
                    'verb': item['verb'],
                }
                elems = item['elements']
                if len(elems) <= 1:
                    continue
                updated = {}
                for key in elems:
                    updated[self.generalize_keys(key)] = self.select_single_phrase(elems[key], phrases)
                mod['elements'] = updated
                if self.debug:
                    print(s['sentence'])
                    print(item['verb'])
                    self.view_frame_on_image(updated)
                modifiled_frames.append(mod)

            if len(modifiled_frames) == 0:
                continue

            entry = {
                'sentence': s['sentence'],
                'frames': modifiled_frames
            }
            processed_entries.append(entry)

        return processed_entries

    def process_and_save_frames(self):
        processed = self.get_process_frames()
        # sorted = self.sort_frames_by_verbs(processed)
        # self._save_ouput_to_json(sorted)