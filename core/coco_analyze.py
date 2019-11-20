from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import os
import copy
import datetime
import time
from collections import defaultdict

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import random
import numpy as np
import matplotlib.pyplot as plt


def _generate_random_colors_hex(color_nums):
    colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(color_nums)]
    return colors


class COCOApi(COCO):
    def __init__(self, dataset, annotation_file=None):
        super(COCOApi, self).__init__(annotation_file=annotation_file)
        self.dataset = dataset
        self.createIndex()


class COCOAnalyze(COCOeval):
    def __init__(self, coco_gt, coco_dt, out_dir, iouType='bbox'):
        super(COCOAnalyze, self).__init__(coco_gt, coco_dt, iouType=iouType)

        if 'bbox' != iouType:
            raise ValueError("Suggested iouType: bbox, but got %s." % iouType)

        self.m_out_dir = out_dir

    def _prepare(self):
        '''
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        '''
        def _toMask(anns, coco):
            # modify ann['segmentation'] by reference
            for ann in anns:
                rle = coco.annToRLE(ann)
                ann['segmentation'] = rle
        p = self.params
        if p.useCats:
            gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
            dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
        else:
            gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds)) #什么类型返回值？
            dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))

        # convert ground truth to mask if iouType == 'segm'
        if p.iouType == 'segm':
            _toMask(gts, self.cocoGt)
            _toMask(dts, self.cocoDt)
        # set ignore flag
        for gt in gts:
            gt['ignore'] = gt['ignore'] if 'ignore' in gt else 0
            gt['ignore'] = gt['ignore'] if gt['ignore'] else ('iscrowd' in gt) and (gt['iscrowd'])
            if p.iouType == 'keypoints':
                gt['ignore'] = (gt['num_keypoints'] == 0) or gt['ignore']
        self._gts = defaultdict(list)       # gt for evaluation  #是一个dict 当元素不存在时返回特定值，其他时候就是一个dict
        self._dts = defaultdict(list)       # dt for evaluation
        for gt in gts:
            self._gts[gt['image_id'], gt['category_id']].append(gt)
        for dt in dts:
            self._dts[dt['image_id'], dt['category_id']].append(dt)
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results
        self.eval     = {}                  # accumulated evaluation results

    def evaluateImg(self, imgId, catId, aRng, maxDet):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]

        if len(gt) == 0 and len(dt) == 0:
            return None

        for g in gt:
            if g['ignore'] or (g['area'] < aRng[0] or g['area'] > aRng[1]):
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o['iscrowd']) for o in gt]
        # load computed ious
        ious = self.ious[imgId, catId][:, gtind] if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm = np.zeros((T, G))
        dtm = np.zeros((T, D))
        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T, D))
        if not len(ious) == 0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t, 1 - 1e-10])
                    m = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind, gind] > 0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m > -1 and gtIg[m] == 0 and gtIg[gind] == 1:
                            break
                        # continue to next gt unless better match made
                        if ious[dind, gind] < iou:
                            continue
                        # if match successful and best so far, store appropriately
                        iou = ious[dind, gind]
                        m = gind
                    # if match made store id of match for both dt and gt
                    if m == -1:
                        continue
                    dtIg[tind, dind] = gtIg[m]
                    dtm[tind, dind] = gt[m]['id']
                    gtm[tind, m] = d['id']
        # set unmatched detections outside of area range to ignore
        a = np.array([d['area'] < aRng[0] or d['area'] > aRng[1] for d in dt]).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm == 0, np.repeat(a, T, 0)))
        # store results for given image and category
        return {
            'image_id': imgId,
            'category_id': catId,
            'aRng': aRng,
            'maxDet': maxDet,
            'dtIds': [d['id'] for d in dt],
            'gtIds': [g['id'] for g in gt],
            'dtMatches': dtm,
            'gtMatches': gtm,
            'dtScores': [d['score'] for d in dt],
            'gtIgnore': gtIg,
            'dtIgnore': dtIg,
        }

    def accumulate(self, p = None):
        '''
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''
        print('Accumulating evaluation results...')
        tic = time.time()
        if not self.evalImgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params
        catIds = p.catIds if p.useCats == 1 else [-1]
        T           = len(p.iouThrs)
        R           = len(p.recThrs)
        K           = len(catIds)
        A           = len(p.areaRng)
        M           = len(p.maxDets)
        precision   = -np.ones((T,R,K,A,M)) # -1 for the precision of absent categories
        recall      = -np.ones((T,K,A,M))
        scores      = -np.ones((T,R,K,A,M))

        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(catIds)  if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
        i_list = [n for n, i in enumerate(p.imgIds)  if i in setI]
        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0*A0*I0
            for a, a0 in enumerate(a_list):
                Na = a0*I0
                for m, maxDet in enumerate(m_list):
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if not e is None]
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])

                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = np.argsort(-dtScores, kind='mergesort')
                    dtScoresSorted = dtScores[inds]

                    dtm  = np.concatenate([e['dtMatches'][:,0:maxDet] for e in E], axis=1)[:,inds]
                    dtIg = np.concatenate([e['dtIgnore'][:,0:maxDet]  for e in E], axis=1)[:,inds]
                    gtIg = np.concatenate([e['gtIgnore'] for e in E])
                    npig = np.count_nonzero(gtIg==0 )
                    if npig == 0:
                        continue
                    tps = np.logical_and(               dtm,  np.logical_not(dtIg) )
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg) )

                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / npig
                        pr = tp / (fp+tp+np.spacing(1))
                        q  = np.zeros((R,))
                        ss = np.zeros((R,))

                        if nd:
                            recall[t,k,a,m] = rc[-1]
                        else:
                            recall[t,k,a,m] = 0

                        # numpy is slow without cython optimization for accessing elements
                        # use python array gets significant speed improvement
                        pr = pr.tolist(); q = q.tolist()

                        for i in range(nd-1, 0, -1):
                            if pr[i] > pr[i-1]:
                                pr[i-1] = pr[i]

                        inds = np.searchsorted(rc, p.recThrs, side='left')
                        try:
                            for ri, pi in enumerate(inds):
                                q[ri] = pr[pi]
                                ss[ri] = dtScoresSorted[pi]
                        except:
                            pass
                        precision[t,:,k,a,m] = np.array(q)
                        scores[t,:,k,a,m] = np.array(ss)
        self.eval = {
            'params': p,
            'counts': [T, R, K, A, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision': precision,
            'recall':   recall,
            'scores': scores,
        }
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format( toc-tic))

    def _make_pr_scores_plot(self, rec_thrs, precisions, scores, name):
        """
        绘制precision根据score变化的曲线
        :param rec_thrs:
        :param precisions:
        :param scores:
        :param name:
        :return:
        """
        cat_ids = self.cocoGt.getCatIds()
        cat_names = [cat['name'] for cat in self.cocoGt.loadCats(ids=cat_ids)]
        # colors = [(1, 1, 1), (.51, .90, .30), (.31, .51, .74),
        #           (.75, .31, .30), (.36, .90, .38), (.50, .39, .64), (1, .6, 0)]
        colors = _generate_random_colors_hex(len(cat_ids))
        area_names = ['all', 'small', 'medium', 'large']
        fig, axes = plt.subplots(2, 2, figsize=(22, 16))

        axes = axes.flatten()
        # plot for per area
        for area_id in range(len(area_names)):
            ax = axes[area_id]
            ax.set_xlim(0, 1.0)
            ax.set_ylim(0, 1.0)
            ax.set_xticks(np.linspace(0, 1.0, 11, endpoint=True))
            ax.set_yticks(np.linspace(0, 1.0, 11, endpoint=True))
            ax.set_xlabel('scores', fontsize='xx-large')
            ax.set_ylabel('precision', fontsize='xx-large')
            ax.set_title('%s-%s' % (name, area_names[area_id]), fontsize='xx-large')

            ax2 = ax.twinx()
            ax2.set_ylabel('recall', fontsize='xx-large')

            for class_id in range(len(cat_ids)):
                per_cat_precisions = precisions[:, class_id, area_id]
                per_cat_scores = scores[:, class_id, area_id]
                ax.plot(per_cat_scores, per_cat_precisions, label=cat_names[class_id])
                ax2.plot(per_cat_scores, rec_thrs)
                # , color=colors[class_id]

            ax.legend(loc='lower left', fontsize='xx-large')

        plt.savefig(os.path.join(self.m_out_dir, "%s-precisions-scores.png" % name))

    def _make_pr_plot(self, rec_thrs, precisons, name, scores=None):
        """
        画PR图
        :param rec_thrs: recall threshold,
        :param precisons: T*R*A, T: iou thresholds; R: recall thresholds; A: areaRngs
        :param name: category name
        :param scores:
        :return:
        """
        colors = [(1, 1, 1), (1, 1, 1), (.31, .51, .74),
                  (.75, .31, .30), (.36, .90, .38), (.50, .39, .64), (1, .6, 0)]
        area_names = ['all','small','medium','large']

        legends =['C75', 'C50', 'Loc', 'Sim', 'Oth', 'BG', 'FN']

        fig, axes = plt.subplots(2, 2, figsize=(22, 16))
        # fig, axes = plt.subplots(1, 1, figsize=(22, 16))
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        else:
            axes = axes.flatten()
        # plot for per area
        for area_id in range(len(area_names)):
            ax = axes[area_id]
            ps = precisons[:, :, area_id]
            ax.set_xlim(0, 1.0)
            ax.set_ylim(0, 1.0)
            ax.set_xticks(np.linspace(0, 1.0, 21, endpoint=True))
            ax.set_yticks(np.linspace(0, 1.0, 21, endpoint=True))
            ax.set_xlabel('recall', fontsize='xx-large')
            ax.set_ylabel('precision', fontsize='xx-large')
            ax.set_title('%s-%s' % (name, area_names[area_id]), fontsize='xx-large')

            if scores is not None:
                ax1 = ax.twiny()
                ax1.set_xlim(0, 1.0)
                ax1.set_xticks(np.linspace(0, 1.0, 21, endpoint=True))
                ax1.set_xlabel('score', fontsize='xx-large')

            # plot for per pr
            for pr_id in range(ps.shape[0]):
                pre = 0
                if pr_id >= 1:
                    pre = ps[pr_id - 1, :]

                cur = ps[pr_id, :]
                ap = np.mean(cur)

                label = "%.3f %s" % (ap, legends[pr_id])
                if scores is not None:
                    if pr_id < scores.shape[0]:
                        cur_scores = scores[pr_id, :, area_id]
                        ax1.plot(cur_scores, cur, label=legends[pr_id])

                ax.fill_between(rec_thrs, pre, cur, facecolor=colors[pr_id], edgecolor='black', label=label)
            if scores is not None:
                ax1.legend(loc='upper left', fontsize='xx-large')
            ax.legend(loc='lower left', fontsize='xx-large')
        plt.tight_layout()
        plt.savefig(os.path.join(self.m_out_dir, "%s-pr.png" % name))

    def analyze(self):
        cat_ids = self.cocoGt.getCatIds()

        org_gt_data = copy.deepcopy(self.cocoGt.dataset)
        org_dt_data = copy.deepcopy(self.cocoDt.dataset)

        rec_thrs = self.params.recThrs

        self.params.maxDets = [100]
        self.params.catIds = cat_ids
        self.params.iouThrs = [0.75, 0.5, 0.1]

        self.evaluate()
        self.accumulate()

        counts = self.eval['counts']
        counts[0] = 7

        precisons = np.zeros(counts, dtype=np.float)
        scores = np.zeros(counts, dtype=np.float)

        precisons[:3, :, :, :, :] = self.eval['precision']
        scores[:3, :, :, :, :] = self.eval['scores']

        self.params.iouThrs = [0.1]
        self.params.useCats = 0

        super_cats = set()

        # do analysis for per category
        for k in range(len(cat_ids)):
            cat_id = cat_ids[k]
            cat_des = self.cocoGt.loadCats(cat_id)[0]
            # if cat_des['name'] not in ['car', 'bus']:
            #     continue
            #
            # if cat_des['name'] == 'bus':
            #     cat_des['name'] = 'person'
            print('cat_des=',cat_des)
            name = "%s-%s" % (cat_des['supercategory'], cat_des['name'])
            print('Analyzing %s (%d)' % (name, k))

            # select detections for single category only
            dt_data = org_dt_data.copy()
            dt_data['annotations'] = [item for item in dt_data['annotations'] if item['category_id'] == cat_id]
            self.cocoDt = COCOApi(dt_data)

            # compute precision but ignore superclass confusion
            gt_data = copy.deepcopy(org_gt_data)
            annotations = gt_data['annotations']
            supercategory = cat_des['supercategory']
            super_cats.add(supercategory)
            cat_ids_in_supercate = self.cocoGt.getCatIds(supNms=supercategory)
            for i in range(len(annotations)):
                anno = annotations[i]
                if anno['category_id'] in cat_ids_in_supercate and anno['category_id'] != cat_id:
                    annotations[i]['ignore'] = 1

            self.cocoGt = COCOApi(gt_data)

            self.evaluate()
            self.accumulate()

            precisons[3, :, k, :, :] = self.eval['precision'][0, :, 0, :, :]
            scores[3, :, k, :, :] = self.eval['scores'][0, :, 0, :, :]

            # compute precison but ignore any class confusion
            for i in range(len(annotations)):
                anno = annotations[i]
                if anno['category_id'] != cat_id:
                    annotations[i]['ignore'] = 1

            gt_data['annotations'] = annotations
            self.cocoGt = COCOApi(gt_data)
            self.evaluate()
            self.accumulate()

            precisons[4, :, k, :, :] = self.eval['precision'][0, :, 0, :, :]
            scores[4, :, k, :, :] = self.eval['scores'][0, :, 0, :, :]
            # fill in background and false negative errors and plot
            precisons[np.where(precisons == -1)] = 0
            precisons[5, :, k, :, :] = precisons[4, :, k, :, :] > 0
            precisons[6, :, k, :, :] = 1

            # make plot for current category
            ps = precisons[:, :, k, :, 0]
            ss = scores[:4, :, k, :, 0]
            # self._make_pr_plot(rec_thrs, ps, name, scores=None)
            self._make_pr_plot(rec_thrs, ps, name, scores=ss)
        
        # plot averages over all categories
        name = "overall-all"
        ps = np.mean(precisons, axis=2)[:, :, :, 0]
        self._make_pr_plot(rec_thrs, ps, name)

        # plot averages over super categories
        # 加载所有的 super categories
        for super_cat in super_cats:
            cat_ids_in_supercate = self.cocoGt.getCatIds(supNms=super_cat)
            valid_inds = np.array([i for i, cat_id in enumerate(cat_ids) if cat_id in cat_ids_in_supercate])

            if len(valid_inds) == 0:
                continue

            super_cat_precisions = precisons[:, :, valid_inds, :, :]

            if super_cat_precisions.ndim == 5:
                ps = np.mean(super_cat_precisions, axis=2)[:, :, :, 0]
            else:
                ps = super_cat_precisions[:, :, :, 0]

            name = "overall-%s" % super_cat
            self._make_pr_plot(rec_thrs, ps, name)

        # # plot precisions vs scores
        # precisons_iou50 = precisons[1, :, :, :, 0]
        # scores_iou50 = scores[1, :, :, :, 0]
        # name = 'IOU50'
        # self._make_pr_scores_plot(rec_thrs, precisons_iou50, scores_iou50, name)


