import os
import torch
from pathlib import Path
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import cv2

from mmrotate.structures import RotatedBoxes
from mmdet.models.utils import samplelist_boxtype2tensor
from mmengine.runner import load_checkpoint
from utils import show_box, show_mask
import matplotlib.pyplot as plt
from mmengine.structures import InstanceData
from data import build_visualizer


@torch.no_grad()
def single_sample_step(i, data, model, predictor, evaluator, dataloader, device, SHOW):
    copied_data = deepcopy(data)  # for sam
    for item in data.values():
        item[0].to(device)

    # Stage 1
    # data['inputs'][0] = torch.flip(data['inputs'][0], dims=[0])
    with torch.no_grad():
        pred_results = model.test_step(data)
    pred_r_bboxes = pred_results[0].pred_instances.bboxes
    pred_r_bboxes = RotatedBoxes(pred_r_bboxes)
    h_bboxes = pred_r_bboxes.convert_to('hbox').tensor
    labels = pred_results[0].pred_instances.labels
    scores = pred_results[0].pred_instances.scores

    # Stage 2
    r_bboxes = []
    if len(h_bboxes) == 0:
        qualities = h_bboxes[:, 0]
        masks = h_bboxes.new_tensor((0, *data['inputs'][0].shape[:2]))
        data_samples = data['data_samples']
    else:
        img = copied_data['inputs'][0].permute(1, 2, 0).numpy()[:, :, ::-1]
        data_samples = copied_data['data_samples']
        data_sample = data_samples[0]
        data_sample = data_sample.to(device=device)

        predictor.set_image(img)
        transformed_boxes = predictor.transform.apply_boxes_torch(h_bboxes, img.shape[:2])
        masks, qualities, lr_logits = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False)
        masks = masks.squeeze(1)
        qualities = qualities.squeeze(-1)
        for mask in masks:
            r_bboxes.append(mask2rbox(mask.cpu().numpy()))

    results_list = get_instancedata_resultlist(r_bboxes, qualities, labels, masks)
    data_samples = add_pred_to_datasample(results_list, data_samples)

    evaluator.process(data_samples=data_samples, data_batch=data)

    if SHOW:
        if len(h_bboxes) != 0:
            show_results(img, masks, h_bboxes, results_list, i, dataloader)

    return evaluator


def mask2rbox(mask):
    y, x = np.nonzero(mask)
    points = np.stack([x, y], axis=-1)
    (cx, cy), (w, h), a = cv2.minAreaRect(points)
    r_bbox = np.array([cx, cy, w, h, a / 180 * np.pi])
    return r_bbox

def show_results(img, masks, h_bboxes, results_list, i, dataloader):
    output_dir = './output_vis/'
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    results = results_list[0]
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    for mask in masks:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    for box in h_bboxes:
        show_box(box.cpu().numpy(), plt.gca())
    plt.axis('off')
    # plt.show()
    plt.savefig(f'./out_mask_{i}.png')
    plt.close()

    # draw rbox with mmrotate
    visualizer = build_visualizer()
    visualizer.dataset_meta = dataloader.dataset.metainfo
    out_img = visualizer._draw_instances(
        img, results,
        dataloader.dataset.metainfo['classes'],
        dataloader.dataset.metainfo['palette'])
    # visualizer.show()
    cv2.imwrite(os.path.join(output_dir, f'out_rbox_{i}.png'),
                out_img[:, :, ::-1])


def add_pred_to_datasample(results_list, data_samples):
    for data_sample, pred_instances in zip(data_samples, results_list):
        data_sample.pred_instances = pred_instances
    samplelist_boxtype2tensor(data_samples)
    return data_samples


def get_instancedata_resultlist(r_bboxes, qualities, labels, masks):
    results = InstanceData()
    results.bboxes = RotatedBoxes(r_bboxes)
    results.scores = qualities
    results.labels = labels
    results.masks = masks.cpu().numpy()
    results_list = [results]
    return results_list
