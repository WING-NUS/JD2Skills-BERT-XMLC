r"""Functional interface"""
import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score, classification_report

__call__ = ['Accuracy','AUC','F1Score','EntityScore','ClassReport','MultiLabelReport','AccuracyThresh']

class Metric:
    def __init__(self):
        self.epsilon = 1.0e-4
        pass

    def __call__(self, outputs, target):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError

class Accuracy(Metric):
    def __init__(self,topK):
        super(Accuracy,self).__init__()
        self.topK = topK
        self.reset()

    def __call__(self, logits, target):
        _, pred = logits.topk(self.topK, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        self.correct_k = correct[:self.topK].view(-1).float().sum(0)
        self.total = target.size(0)

    def reset(self):
        self.correct_k = 0
        self.total = 0

    def value(self):
        return float(self.correct_k)  / self.total

    def name(self):
        return 'accuracy'


class AccuracyThresh(Metric):
    def __init__(self,thresh = 0.5):
        super(AccuracyThresh,self).__init__()
        self.thresh = thresh
        self.reset()

    def __call__(self, logits, target):
        self.y_pred = logits.sigmoid()
        self.y_true = target

    def reset(self):
        self.correct_k = 0
        self.total = 0

    def value(self):
        data_size = self.y_pred.size(0)
        acc = np.mean(((self.y_pred>self.thresh).type(torch.DoubleTensor)==self.y_true.type(torch.DoubleTensor)).float().cpu().numpy(), axis=1).sum()
        return acc / data_size

    def name(self):
        return 'accuracy'

class MRR(Metric):
    def __init__(self):
        super(MRR,self).__init__()
        self.rr = []

    def __call__(self, logits, target):
        self.y_pred = logits.sigmoid().cpu().numpy()
        self.y_true = target.cpu().numpy()

        for i in range(len(self.y_pred)):
            y_t = np.where(self.y_true[i]==1)[0]
            y_t_len = len(y_t)
            y_p_index = np.flip(np.argsort(self.y_pred[i]),0)
            for i in range(len(y_p_index)):
                if y_p_index[i] in y_t:
                    self.rr.append(1/float(i+1))
                    break

    def reset(self):
        return

    def value(self):
        return np.mean(np.array(self.rr))*100

    def name(self):
        return 'mrr'

    def show(self):
        print("MRR: {}".format(self.value()))

class NDCG(Metric):
    def __init__(self):
        super(NDCG,self).__init__()
        self.NDCG = []
        self.kvals = [5,10,30,50,100]
        self.reset()

    def __call__(self, logits, target):
        self.y_pred = logits.sigmoid().cpu().numpy()
        self.y_true = target.cpu().numpy()
        
        for i in range(len(self.y_pred)):
            y_t = np.where(self.y_true[i]==1)[0]
            y_t_len = len(y_t)
            y_p_index = np.flip(np.argsort(self.y_pred[i]),0)
            idcg = np.sum([1.0/np.log2(x+2) for x in range(y_t_len)])
            ndcg = []
            for k in self.kvals:
                dcg = 0 
                for i in range(0,k):
                    if y_p_index[i] in y_t:
                        dcg = dcg + 1.0/np.log2(i+2)
                ndcg.append(dcg/idcg)
            self.NDCG.append(ndcg)

    def reset(self):
        return

    def value(self):
        return np.mean(np.array(self.NDCG),axis=0)*100

    def name(self):
        return 'ndcg'

    def show(self):
        idx = 0
        result = self.value()
        for k in self.kvals:
            print("NDCG@{}: {}".format(k,result[idx]))
            idx += 1

class Recall(Metric):
    def __init__(self):
        super(Recall,self).__init__()
        self.Recall = []
        self.kvals = [5,10,30,50,100]
        self.reset()

    def __call__(self, logits, target):
        self.y_pred = logits.sigmoid().cpu().numpy()        
        self.y_true = target.cpu().numpy()
        for i in range(len(self.y_pred)):
            recall = []
            y_p_index = np.flip(np.argsort(self.y_pred[i]),0)
            y_t = np.where(self.y_true[i]==1)[0]
            y_t_len = len(y_t)
            for k in self.kvals:
                correct = len(np.intersect1d(y_t,y_p_index[0:k]))
                recall.append(correct/(y_t_len+self.epsilon))
            self.Recall.append(recall)
        

    def reset(self):
        return

    def value(self):
        return np.mean(np.array(self.Recall),axis=0)*100

    def name(self):
        return 'recall'
    
    def show(self):
        idx = 0
        result = self.value()
        for k in self.kvals:
            print("Recall@{}: {}".format(k,result[idx]))
            idx += 1

class Implicit_Metrics(Metric):
    skill_list = []
    i2w = {}
    i2l = {}

    def __init__(self, skill_list_path, i2w, i2l):
        super(Implicit_Metrics,self).__init__()
        if not self.skill_list:
            self.skill_list = self.read_skills(path=skill_list_path)
        if not self.i2w:
            self.i2w = i2w
        if not self.i2l:
            self.i2l = i2l
    
    def read_skills(self, path):
        skill_list = []
        with open(path, 'r') as f:
            for line in f.readlines():
                skill = line.replace('\n','')
                skill_list.append(skill)
        return skill_list

    def present_skills(self, job_description, skill_list):
        skills = []
        for skill in skill_list:
            if (' ' + skill.lower().strip() + ' ') in (' ' + job_description.lower().strip() + ' '):
                if skill not in skills:
                    skills.append(skill)
        return skills

    def generate_sentence(self, input_ids):
        input_ids = input_ids.tolist()
        sentence = ""
        for idx in input_ids:
            if self.i2w[idx][:2]=="##":
                sentence += self.i2w[idx][2:]
            else:
                sentence += " " + self.i2w[idx]
        return sentence.strip()
    
    def job_labels(self, label_indices):
        labels = []
        for idx in label_indices:
            labels.append(self.i2l[idx])
        return labels

    def intersection(self, lst1, lst2):
        lst3 = [value for value in lst1 if value in lst2]
        return lst3

class EIM(Implicit_Metrics):
    def __init__(self, path, i2w, i2l):
        super(EIM,self).__init__(path, i2w, i2l)
        self.EIM = []
        self.reset()

    def __call__(self, input_ids, output, target):
        self.input_ids = input_ids
        self.y_pred = output.sigmoid().cpu().numpy()        
        self.y_true = target.cpu().numpy()
        for i in range(len(self.y_pred)):
            self.EIM.append(self.eim(self.input_ids[i], self.y_pred[i], self.y_true[i]))
        
            
    def eim(self, input_ids, output, target):
        job_description = self.generate_sentence(input_ids)    
        sorted_prediction_indices = np.flip(np.argsort(output))
        sorted_prediction_indices = sorted_prediction_indices[:20]
        label_indices = np.where(target==1)[0]
        
        job_labels = self.job_labels(label_indices)
        predicted_labels = self.job_labels(sorted_prediction_indices)
        explicit_skills = self.present_skills(job_description, self.skill_list)

        return len(self.intersection(predicted_labels, explicit_skills)) / (len(self.intersection(job_labels, explicit_skills)) + self.epsilon)


    def reset(self):
        return

    def value(self):
        return np.mean(np.array(self.EIM))*100

    def name(self):
        return 'eim'
    
    def show(self):
        result = self.value()
        print("EIM: {}".format(result))

class RIIM(Implicit_Metrics):
    def __init__(self, path, i2w, i2l):
        super(RIIM,self).__init__(path, i2w, i2l)
        self.RIIM = []
        self.reset()

    def __call__(self, input_ids, output, target):
        self.input_ids = input_ids
        self.y_pred = output.sigmoid().cpu().numpy()        
        self.y_true = target.cpu().numpy()
        for i in range(len(self.y_pred)):
            self.RIIM.append(self.riim(self.input_ids[i], self.y_pred[i], self.y_true[i]))
        
    def riim(self, input_ids, output, target):
        job_description = self.generate_sentence(input_ids)
        sorted_prediction_indices = np.flip(np.argsort(output))
        sorted_prediction_indices = sorted_prediction_indices[:20]
        label_indices = np.where(target==1)[0]
        
        job_labels = self.job_labels(label_indices)
        predicted_labels = self.job_labels(sorted_prediction_indices)
        explicit_skills = self.present_skills(job_description, self.skill_list)
        implicit_skills = [val for val in job_labels if val not in explicit_skills]
        predicted_implicit_skills = [val for val in predicted_labels if val in implicit_skills]

        return len(predicted_implicit_skills)/(len(implicit_skills) + self.epsilon)

    def reset(self):
        return

    def value(self):
        return np.mean(np.array(self.RIIM))*100

    def name(self):
        return 'riim'
    
    def show(self):
        result = self.value()
        print("RIIM: {}".format(result))

class REIM(Implicit_Metrics):
    def __init__(self, path, i2w, i2l):
        super(REIM,self).__init__(path, i2w, i2l)
        self.REIM = []
        self.reset()

    def __call__(self, input_ids, output, target):
        self.input_ids = input_ids
        self.y_pred = output.sigmoid().cpu().numpy()        
        self.y_true = target.cpu().numpy()
        for i in range(len(self.y_pred)):
            self.REIM.append(self.reim(self.input_ids[i], self.y_pred[i], self.y_true[i]))
        
            
    def reim(self, input_ids, output, target):
        job_description = self.generate_sentence(input_ids)
        sorted_prediction_indices = np.flip(np.argsort(output))
        sorted_prediction_indices = sorted_prediction_indices[:20]
        label_indices = np.where(target==1)[0]
        
        job_labels = self.job_labels(label_indices)
        predicted_labels = self.job_labels(sorted_prediction_indices)
        explicit_skills = self.present_skills(job_description, self.skill_list)

        return len(self.intersection(predicted_labels, explicit_skills)) / (len(explicit_skills) + self.epsilon)


    def reset(self):
        return

    def value(self):
        return np.mean(np.array(self.REIM))*100

    def name(self):
        return 'reim'
    
    def show(self):
        result = self.value()
        print("REIM: {}".format(result))


class AUC(Metric):
    '''
    AUC score
    micro:
            Calculate metrics globally by considering each element of the label
            indicator matrix as a label.
    macro:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
    weighted:
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label).
    samples:
            Calculate metrics for each instance, and find their average.
    Example:
        >>> metric = AUC(**)
        >>> for epoch in range(epochs):
        >>>     metric.reset()
        >>>     for batch in batchs:
        >>>         logits = model()
        >>>         metric(logits,target)
        >>>         print(metric.name(),metric.value())
    '''

    def __init__(self,task_type = 'binary',average = 'binary'):
        super(AUC, self).__init__()

        assert task_type in ['binary','multiclass']
        assert average in ['binary','micro', 'macro', 'samples', 'weighted']

        self.task_type = task_type
        self.average = average

    def __call__(self,logits,target):
        if self.task_type == 'binary':
            self.y_prob = logits.sigmoid().data.cpu().numpy()
        else:
            self.y_prob = logits.softmax(-1).data.cpu().detach().numpy()
        self.y_true = target.cpu().numpy()

    def reset(self):
        self.y_prob = 0
        self.y_true = 0

    def value(self):
        auc = roc_auc_score(y_score=self.y_prob, y_true=self.y_true, average=self.average)
        return auc

    def name(self):
        return 'auc'

class F1Score(Metric):
    '''
    F1 Score
    binary:
            Only report results for the class specified by ``pos_label``.
            This is applicable only if targets (``y_{true,pred}``) are binary.
    micro:
            Calculate metrics globally by considering each element of the label
            indicator matrix as a label.
    macro:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
    weighted:
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label).
    samples:
            Calculate metrics for each instance, and find their average.
    Example:
        >>> metric = F1Score(**)
        >>> for epoch in range(epochs):
        >>>     metric.reset()
        >>>     for batch in batchs:
        >>>         logits = model()
        >>>         metric(logits,target)
        >>>         print(metric.name(),metric.value())
    '''
    def __init__(self,thresh = 0.5, normalizate = True,task_type = 'binary',average = 'binary',search_thresh = False):
        super(F1Score).__init__()
        assert task_type in ['binary','multiclass']
        assert average in ['binary','micro', 'macro', 'samples', 'weighted']

        self.thresh = thresh
        self.task_type = task_type
        self.normalizate  = normalizate
        self.search_thresh = search_thresh
        self.average = average

    def thresh_search(self,y_prob):
        best_threshold = 0
        best_score = 0
        for threshold in tqdm([i * 0.01 for i in range(100)], disable=True):
            self.y_pred = y_prob > threshold
            score = self.value()
            if score > best_score:
                best_threshold = threshold
                best_score = score
        return best_threshold,best_score

    def __call__(self,logits,target):
        self.y_true = target.cpu().numpy()
        if self.normalizate and self.task_type == 'binary':
            y_prob = logits.sigmoid().data.cpu().numpy()
        elif self.normalizate and self.task_type == 'multiclass':
            y_prob = logits.softmax(-1).data.cpu().detach().numpy()
        else:
            y_prob = logits.cpu().detach().numpy()

        if self.task_type == 'binary':
            if self.thresh and self.search_thresh == False:
                self.y_pred = (y_prob > self.thresh ).astype(int)
                self.value()
            else:
                thresh,f1 = self.thresh_search(y_prob = y_prob)
                print(f"Best thresh: {thresh:.4f} - F1 Score: {f1:.4f}")

        if self.task_type == 'multiclass':
            self.y_pred = np.argmax(y_prob, 1)

    def reset(self):
        self.y_pred = 0
        self.y_true = 0

    def value(self):
        f1 = f1_score(y_true=self.y_true, y_pred=self.y_pred, average=self.average)
        return f1

    def name(self):
        return 'f1'

class ClassReport(Metric):
    '''
    class report
    '''
    def __init__(self,target_names = None):
        super(ClassReport).__init__()
        self.target_names = target_names

    def reset(self):
        self.y_pred = 0
        self.y_true = 0

    def value(self):
        score = classification_report(y_true = self.y_true,
                                      y_pred = self.y_pred,
                                      target_names=self.target_names)
        print(f"\n\n classification report: {score}")

    def __call__(self,logits,target):
        _, y_pred = torch.max(logits.data, 1)
        self.y_pred = y_pred.cpu().numpy()
        self.y_true = target.cpu().numpy()

    def name(self):
        return "class_report"

class MultiLabelReport(Metric):
    '''
    multi label report
    '''
    def __init__(self,id2label = None):
        super(MultiLabelReport).__init__()
        self.id2label = id2label

    def reset(self):
        self.y_prob = 0
        self.y_true = 0

    def __call__(self,logits,target):

        self.y_prob = logits.sigmoid().data.cpu().detach().numpy()
        self.y_true = target.cpu().numpy()

    def value(self):
        for i, label in self.id2label.items():
            auc = roc_auc_score(y_score=self.y_prob[:, i], y_true=self.y_true[:, i])
            print(f"label:{label} - auc: {auc:.4f}")

    def name(self):
        return "multilabel_report"
