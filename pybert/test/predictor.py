#encoding:utf-8
import torch
import numpy as np
from ..common.tools import model_device, parse_idx
from ..callback.progressbar import ProgressBar
from pybert.train.metrics import MRR, Recall, NDCG, EIM, REIM, RIIM
from pybert.configs.basic_config import config

class Predictor(object):
    def __init__(self,
                 model,
                 logger,
                 n_gpu,
                 i2w,
                 i2l
                 ):
        self.model = model
        self.logger = logger
        self.model, self.device = model_device(n_gpu= n_gpu, model=self.model)
        self.i2w = i2w
        self.i2l = i2l

    def predict(self,data):
        pbar = ProgressBar(n_total=len(data))
        all_logits = None
        self.model.eval()
        test_metrics = [MRR(), NDCG(), Recall(), EIM(config['data_label_path'],self.i2w,self.i2l), RIIM(config['data_label_path'],self.i2w,self.i2l), REIM(config['data_label_path'],self.i2w,self.i2l)]
        implicit_metrics = ["eim","riim","reim"]
        with torch.no_grad():
            for step, batch in enumerate(data):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                logits = self.model(input_ids, segment_ids, input_mask)
                
                for metric in test_metrics:
                    if metric.name() in implicit_metrics:
                        metric(input_ids=input_ids, output=logits, target=label_ids)
                    else:
                        metric(logits=logits, target=label_ids)
                
                logits = logits.sigmoid()
                if all_logits is None:
                    all_logits = logits.detach().cpu().numpy()
                else:
                    all_logits = np.concatenate([all_logits,logits.detach().cpu().numpy()],axis = 0)
                pbar.batch_step(step=step,info = {},bar_type='Testing')
        for metric in test_metrics:
                metric.show()
        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()
        return all_logits

    def job_labels(self, label_indices):
        labels = []
        for idx in label_indices:
            labels.append(self.i2l[idx])
        return labels

    def print_labels(self,logits,idx):
        sorted_prediction_indices = np.flip(np.argsort(logits[idx]))
        sorted_prediction_indices = sorted_prediction_indices[:20]
        predicted_labels = self.job_labels(sorted_prediction_indices)
        print("prediction {}: {}".format(idx,predicted_labels))

    def labels(self,logits,idx):
        idx = parse_idx(idx,logits.shape[0])
        print("-"*89)
        print("printing labels")
        for i in idx:
            self.print_labels(logits,i)