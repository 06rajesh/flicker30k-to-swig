
import cv2
from frame_semantic_transformer import FrameSemanticTransformer

from flickr30k_entities_utils import get_annotations, get_sentence_data

def debug_img(imgid):
    sentence_file = 'annotations/Sentences/{}.txt'.format(imgid)
    sentence = get_sentence_data(sentence_file)
    print(sentence)

    annotation_file = 'annotations/Annotations/{}.xml'.format(imgid)
    annotation = get_annotations(annotation_file)
    # print(annotation)

    img_file = 'flickr30k-images/{}.jpg'.format(imgid)
    # read image
    img = cv2.imread(img_file)

    result = img.copy()
    boxes = annotation['boxes']
    for key in boxes:
        for b in boxes[key]:
            cv2.rectangle(result, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)

    # show thresh and result
    cv2.imshow("bounding_box", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    imgid = 4584267739
    sentence_file = 'annotations/Sentences/{}.txt'.format(imgid)
    sentences = get_sentence_data(sentence_file)

    frame_transformer = FrameSemanticTransformer("base")

    for s in sentences:
        # v result = frame_transformer.detect_frames(s['sentence'])
        # first detect trigger locations
        base_sentence, trigger_locs = frame_transformer._identify_triggers(s['sentence'])

        print(base_sentence)
        print(trigger_locs)

        print("=========================")

        # print(f"Results found in: {result.sentence}")
        # for frame in result.frames:
        #     print(f"FRAME: {frame.name}")
        #     for element in frame.frame_elements:
        #         print(f"{element.name}: {element.text}")
        #     print("=========================")
        #
        # print("=============================")
        # print("\n")

    #
    # # debug_img('110820212')
