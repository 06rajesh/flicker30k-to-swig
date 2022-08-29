import os
import re
import json
from typing import List

from pathlib import Path
import nltk
import spacy
import pyinflect
from spacy import Language
from nltk.stem import SnowballStemmer
from frame_semantic_transformer import FrameSemanticTransformer, DetectFramesResult, FrameResult, FrameElementResult
from flickr30k_entities_utils import get_sentence_data

class FlickerSentenceSwigFramer:
    stemmer: SnowballStemmer
    frame_parser: FrameSemanticTransformer
    sentence_dir: Path
    export_dir: Path
    spacy_nlp: Language
    be_verbs: List[str]

    def __init__(self, sentences_dir:str, export_dir:str):
        self.stemmer = SnowballStemmer(language='english')
        self.frame_parser = FrameSemanticTransformer("base")

        self.sentence_dir = Path(sentences_dir)
        self.export_dir = Path(export_dir)

        self.spacy_nlp = spacy.load("en_core_web_sm")

        self.be_verbs = ['am', 'are', 'is', 'was', 'were', 'been', 'being']

        if not os.path.exists(self.export_dir):
            os.mkdir(self.export_dir)

    def get_word_index(self, s, idx):
        words = re.findall(r'\s*\S+\s*', s)
        return sum(map(len, words[:idx])) + len(words[idx]) - len(words[idx].lstrip())

    def get_words_pos_list(self, s: str, words: List[str]):
        start = 0
        strlen = len(s)
        positions = []
        for w in words:
            spart = s[start:strlen]
            wpos = spart.find(w)
            if wpos != -1:
                positions.append(start + wpos)
                start = start + wpos + len(w)
        return positions

    def get_vbg_by_stemming(self, verb:str):
        vowels = ['a', 'e', 'i', 'o', 'u']

        root = self.stemmer.stem(verb)
        if root[-1] in vowels:
            root = root[:-1]

        framename = root + 'ing'
        return framename

    def get_swig_frames_from_verb(self, verb:str):
        framename = verb
        if not verb.endswith('ing'):
            doc_dep = self.spacy_nlp(verb)
            token = doc_dep[0]
            vbg = token._.inflect("VBG")
            if not vbg:
                vbg = self.get_vbg_by_stemming(verb)
            framename = vbg

        return framename

    def get_nltk_verbs(self, sentence:str):
        text = nltk.word_tokenize(sentence)
        pos_tagged = nltk.pos_tag(text)

        nltk_verbs = [pos_tuple[0] for pos_tuple in pos_tagged if pos_tuple[1].startswith('VB') and pos_tuple[0] not in self.be_verbs]
        return nltk_verbs

    def get_spacy_verbs(self, sentence:str):
        doc = self.spacy_nlp(sentence)
        spacy_verbs = [token.text for token in doc if token.pos_ == "VERB"]
        return spacy_verbs

    def get_verb_idx(self, sentence: str, with_frames:bool=False):

        # spacy_verbs = self.get_spacy_verbs(sentence)
        # print(spacy_verbs)

        verbs = self.get_spacy_verbs(sentence)
        verbs_pos = self.get_words_pos_list(sentence, verbs)

        if with_frames:
            frames_list = [self.get_swig_frames_from_verb(v) for v in verbs]
            return verbs_pos, frames_list

        return verbs_pos

    def detect_frames_with_custom_locs(self, sentence: str, custom_trigger_locs: List[int]) -> DetectFramesResult:
        # next detect frames for each trigger
        frames = self.frame_parser._classify_frames(sentence, custom_trigger_locs)

        frame_and_locs = [
            (frame, loc) for frame, loc in zip(frames, custom_trigger_locs) if frame
        ]
        frame_elements_lists = self.frame_parser._extract_frame_args(sentence, frame_and_locs)
        frame_results: list[FrameResult] = []
        for ((frame, loc), frame_element_tuples) in zip(
                frame_and_locs, frame_elements_lists
        ):
            frame_elements = [
                FrameElementResult(element, text)
                for element, text in frame_element_tuples
            ]
            frame_results.append(
                FrameResult(
                    name=frame,
                    trigger_location=loc,
                    frame_elements=frame_elements,
                )
            )
        return DetectFramesResult(
            sentence,
            trigger_locations=custom_trigger_locs,
            frames=frame_results,
        )

    def _read_sentences_by_id(self, img_id):
        sentence_file = self.sentence_dir / '{}.txt'.format(img_id)
        return get_sentence_data(sentence_file)

    def _save_ouput_to_json(self, output, img_id):
        outfile = self.export_dir / '{}.json'.format(img_id)
        with open(outfile, 'w') as f:
            json.dump(output, f)

    def export_sentence_frames(self, img_id:str):
        sentences = self._read_sentences_by_id(img_id)
        output = []

        for s in sentences:
            base_sentence, _ = self.frame_parser._identify_triggers(s['sentence'])
            verb_idx, verbs = self.get_verb_idx(base_sentence, with_frames=True)

            result = self.detect_frames_with_custom_locs(base_sentence, verb_idx)
            frames = []
            for idx, frame in enumerate(result.frames):
                frame_out = {
                    'frame': frame.name,
                    'verb': verbs[idx],
                }
                elements = {}
                for element in frame.frame_elements:
                    elements[element.name.lower()] = element.text
                frame_out['elements'] = elements

                frames.append(frame_out)

            sout = {'sentence': base_sentence, 'frames': frames}
            output.append(sout)

        self._save_ouput_to_json(output, img_id)