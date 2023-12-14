import os
import sys
sys.path.append(r"E:\PyCharm Projects\Pedestron-Caltech-master")
from tools.cityPerson.coco import COCO
from tools.cityPerson.eval_MR_multisetup import COCOeval


def validate(annFile, dt_path):
    mean_MR = []
    my_id_setup = []
    for id_setup in range(0, 4):  # /*** different index of subsets of dataset: reasonable, small, heavy and all ***/
        cocoGt = COCO(annFile)   # ground truth annotation
        cocoDt = cocoGt.loadRes(dt_path)  # detections/prediction results
        imgIds = sorted(cocoGt.getImgIds())  # image id
        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')  # initialize evaluations
        cocoEval.params.imgIds = imgIds
        # execute evaluations per image per category (return dtMatches, gtMatches, dtScores)
        cocoEval.evaluate(id_setup)
        cocoEval.accumulate()       # accumulate detection results per image per category into a dictionary
        mean_MR.append(cocoEval.summarize_nofile(id_setup))  # calculate Log-Average Missing Rate
        my_id_setup.append(id_setup)
    return mean_MR


def validate_with_output(annFile, dt_path, res_file_pointer):
    mean_MR = []
    my_id_setup = []
    for id_setup in range(0, 4):  # /*** different index of subsets of dataset: reasonable, small, heavy and all ***/
        cocoGt = COCO(annFile)   # ground truth annotation
        cocoDt = cocoGt.loadRes(dt_path)  # detections/prediction results
        imgIds = sorted(cocoGt.getImgIds())  # image id
        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')  # initialize evaluations
        cocoEval.params.imgIds = imgIds
        # execute evaluations per image per category (return dtMatches, gtMatches, dtScores)
        cocoEval.evaluate(id_setup)
        cocoEval.accumulate()       # accumulate detection results per image per category into a dictionary
        #mean_MR.append(cocoEval.summarize_nofile(id_setup))  # calculate Log-Average Missing Rate
        mean_MR.append(cocoEval. summarize(id_setup, res_file_pointer))  # calculate Log-Average Missing Rate
        my_id_setup.append(id_setup)
    return mean_MR
