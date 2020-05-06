import pandas as pd
import os
import numpy as np

try:
    import torch
except ImportError:
    torch = None

### Evaluator for link property prediction
class Evaluator:
    def __init__(self, name):
        self.name = name

        meta_info = pd.read_csv(os.path.join(os.path.dirname(__file__), "master.csv"), index_col = 0)
        if not self.name in meta_info:
            print(self.name)
            error_mssg = "Invalid dataset name {}.\n".format(self.name)
            error_mssg += "Available datasets are as follows:\n"
            error_mssg += "\n".join(meta_info.keys())
            raise ValueError(error_mssg)

        self.eval_metric = meta_info[self.name]["eval metric"]

        if "hits@" in self.eval_metric:
            ### Hits@K

            self.K = int(self.eval_metric.split('@')[1])


    def _parse_and_check_input(self, input_dict):
        if "hits@" in self.eval_metric:
            if not "y_pred_pos" in input_dict:
                RuntimeError("Missing key of y_pred_pos")
            if not "y_pred_neg" in input_dict:
                RuntimeError("Missing key of y_pred_neg")

            y_pred_pos, y_pred_neg = input_dict["y_pred_pos"], input_dict["y_pred_neg"]

            """
                y_pred_pos: numpy ndarray or torch tensor of shape (num_edge, )
                y_pred_neg: numpy ndarray or torch tensor of shape (num_edge, )
            """

            # convert y_pred_pos, y_pred_neg into either torch tensor or both numpy array
            # type_info stores information whether torch or numpy is used

            type_info = None

            # check the raw tyep of y_pred_pos
            if not (isinstance(y_pred_pos, np.ndarray) or (torch is not None and isinstance(y_pred_pos, torch.Tensor))):
                raise ValueError("y_pred_pos needs to be either numpy ndarray or torch tensor")

            # check the raw type of y_pred_neg
            if not (isinstance(y_pred_neg, np.ndarray) or (torch is not None and isinstance(y_pred_neg, torch.Tensor))):
                raise ValueError("y_pred_neg needs to be either numpy ndarray or torch tensor")

            # if either y_pred_pos or y_pred_neg is torch tensor, use torch tensor
            if torch is not None and (isinstance(y_pred_pos, torch.Tensor) or isinstance(y_pred_neg, torch.Tensor)):
                # converting to torch.Tensor to numpy on cpu
                if isinstance(y_pred_pos, np.ndarray):
                    y_pred_pos = torch.from_numpy(y_pred_pos)

                if isinstance(y_pred_neg, np.ndarray):
                    y_pred_neg = torch.from_numpy(y_pred_neg)

                # put both y_pred_pos and y_pred_neg on the same device
                y_pred_pos = y_pred_pos.to(y_pred_neg.device)

                type_info = 'torch'

            else:
                # both y_pred_pos and y_pred_neg are numpy ndarray

                type_info = 'numpy'


            if not y_pred_pos.ndim == 1:
                raise RuntimeError("y_pred_pos must to 1-dim arrray, {}-dim array given".format(y_pred_pos.ndim))

            if not y_pred_neg.ndim == 1:
                raise RuntimeError("y_pred_neg must to 1-dim arrray, {}-dim array given".format(y_pred_neg.ndim))

            return y_pred_pos, y_pred_neg, type_info

        elif 'mrr' == self.eval_metric:

            if not "y_pred_pos" in input_dict:
                RuntimeError("Missing key of y_pred_pos")
            if not "y_pred_neg" in input_dict:
                RuntimeError("Missing key of y_pred_neg")

            y_pred_pos, y_pred_neg = input_dict["y_pred_pos"], input_dict["y_pred_neg"]

            """
                y_pred_pos: numpy ndarray or torch tensor of shape (num_edge, )
                y_pred_neg: numpy ndarray or torch tensor of shape (num_edge, num_node_negative)
            """

            # convert y_pred_pos, y_pred_neg into either torch tensor or both numpy array
            # type_info stores information whether torch or numpy is used

            type_info = None

            # check the raw tyep of y_pred_pos
            if not (isinstance(y_pred_pos, np.ndarray) or (torch is not None and isinstance(y_pred_pos, torch.Tensor))):
                raise ValueError("y_pred_pos needs to be either numpy ndarray or torch tensor")

            # check the raw type of y_pred_neg
            if not (isinstance(y_pred_neg, np.ndarray) or (torch is not None and isinstance(y_pred_neg, torch.Tensor))):
                raise ValueError("y_pred_neg needs to be either numpy ndarray or torch tensor")

            # if either y_pred_pos or y_pred_neg is torch tensor, use torch tensor
            if torch is not None and (isinstance(y_pred_pos, torch.Tensor) or isinstance(y_pred_neg, torch.Tensor)):
                # converting to torch.Tensor to numpy on cpu
                if isinstance(y_pred_pos, np.ndarray):
                    y_pred_pos = torch.from_numpy(y_pred_pos)

                if isinstance(y_pred_neg, np.ndarray):
                    y_pred_neg = torch.from_numpy(y_pred_neg)

                # put both y_pred_pos and y_pred_neg on the same device
                y_pred_pos = y_pred_pos.to(y_pred_neg.device)

                type_info = 'torch'


            else:
                # both y_pred_pos and y_pred_neg are numpy ndarray

                type_info = 'numpy'


            if not y_pred_pos.ndim == 1:
                raise RuntimeError("y_pred_pos must to 1-dim arrray, {}-dim array given".format(y_pred_pos.ndim))

            if not y_pred_neg.ndim == 2:
                raise RuntimeError("y_pred_neg must to 2-dim arrray, {}-dim array given".format(y_pred_neg.ndim))

            return y_pred_pos, y_pred_neg, type_info

        else:
            raise ValueError("Undefined eval metric %s" % (self.eval_metric))


    def eval(self, input_dict):

        if "hits@" in self.eval_metric:
            y_pred_pos, y_pred_neg, type_info = self._parse_and_check_input(input_dict)
            return self._eval_hits(y_pred_pos, y_pred_neg, type_info)
        elif self.eval_metric == 'mrr':
            y_pred_pos, y_pred_neg, type_info = self._parse_and_check_input(input_dict)
            return self._eval_mrr(y_pred_pos, y_pred_neg, type_info)
            
        else:
            raise ValueError("Undefined eval metric %s" % (self.eval_metric))

    @property
    def expected_input_format(self):
        desc = "==== Expected input format of Evaluator for {}\n".format(self.name)
        if "hits@" in self.eval_metric:
            desc += "{\"y_pred_pos\": y_pred_pos, \"y_pred_neg\": y_pred_neg}\n"
            desc += "- y_pred_pos: numpy ndarray or torch tensor of shape (num_edge, ). Torch tensor on GPU is recommended for efficiency.\n"
            desc += "- y_pred_neg: numpy ndarray or torch tensor of shape (num_edge, ). Torch tensor on GPU is recommended for efficiency.\n"
            desc += "y_pred_pos is the predicted scores for positive edges.\n"
            desc += "y_pred_neg is the predicted scores for negative edges.\n"
            desc += "Note: As the evaluation metric is ranking-based, the predicted scores need to be different for different edges."
        elif self.eval_metric == "mrr":
            desc += "{\"y_pred_pos\": y_pred_pos, \"y_pred_neg\": y_pred_neg}\n"
            desc += "- y_pred_pos: numpy ndarray or torch tensor of shape (num_edge, ). Torch tensor on GPU is recommended for efficiency.\n"
            desc += "- y_pred_neg: numpy ndarray or torch tensor of shape (num_edge, num_nodes_neg). Torch tensor on GPU is recommended for efficiency.\n"
            desc += "y_pred_pos is the predicted scores for positive edges.\n"
            desc += "y_pred_neg is the predicted scores for negative edges. It needs to be a 2d matrix.\n"
            desc += "y_pred_pos[i] is ranked among y_pred_neg[i].\n"
            desc += "Note: As the evaluation metric is ranking-based, the predicted scores need to be different for different edges."
        else:
            raise ValueError("Undefined eval metric %s" % (self.eval_metric))

        return desc

    @property
    def expected_output_format(self):
        desc = "==== Expected output format of Evaluator for {}\n".format(self.name)
        if "hits@" in self.eval_metric:
            desc += "{" + "hits@{}\": hits@{}".format(self.K, self.K) + "}\n"
            desc += "- hits@{} (float): Hits@{} score\n".format(self.K, self.K)
        elif self.eval_metric == 'mrr':
            desc += "{" + "\"hits@1_list\": hits@1_list, \"hits@3_list\": hits@3_list, \n\"hits@10_list\": hits@10_list, \"mrr_list\": mrr_list}\n"
            desc += "- mrr_list (list of float): list of scores for calculating MRR \n"
            desc += "- hits@1_list (list of float): list of scores for calculating Hits@1 \n" 
            desc += "- hits@3_list (list of float): list of scores to calculating Hits@3\n"
            desc += "- hits@10_list (list of float): list of scores to calculating Hits@10\n" 
            desc += "Note: i-th element corresponds to the prediction score for the i-th edge.\n" 
            desc += "Note: To obtain the final score, you need to concatenate the lists of scores and take average over the concatenated list."
        else:
            raise ValueError("Undefined eval metric %s" % (self.eval_metric))

        return desc


    def _eval_hits(self, y_pred_pos, y_pred_neg, type_info):
        """
            compute Hits@K
            For each positive target node, the negative target nodes are the same.

            y_pred_neg is an array.
            rank y_pred_pos[i] against y_pred_neg for each i
        """

        if len(y_pred_neg) < self.K:
            return {"hits@{}".format(self.K): 1.}

        if type_info == 'torch':
            kth_score_in_negative_edges = torch.topk(y_pred_neg, self.K)[0][-1]
            hitsK = float(torch.sum(y_pred_pos > kth_score_in_negative_edges).cpu()) / len(y_pred_pos)

        # type_info is numpy
        else:
            kth_score_in_negative_edges = np.sort(y_pred_neg)[-self.K]
            hitsK = float(np.sum(y_pred_pos > kth_score_in_negative_edges)) / len(y_pred_pos)

        return {"hits@{}".format(self.K): hitsK}
    
    def _eval_mrr(self, y_pred_pos, y_pred_neg, type_info):
        """
            compute mrr
            y_pred_neg is an array with shape (batch size, num_entities_neg).
            y_pred_pos is an array with shape (batch size, )
        """


        if type_info == 'torch':
            y_pred = torch.cat([y_pred_pos.view(-1,1), y_pred_neg], dim = 1)
            argsort = torch.argsort(y_pred, dim = 1, descending = True)
            ranking_list = (argsort == 0).nonzero()
            ranking_list = ranking_list[:, 1] + 1
            hits1_list = (ranking_list <= 1).to(torch.float)
            hits3_list = (ranking_list <= 3).to(torch.float)
            hits10_list = (ranking_list <= 10).to(torch.float)
            mrr_list = 1./ranking_list.to(torch.float)

            return {"hits@1_list": hits1_list, 
                     "hits@3_list": hits3_list,
                     "hits@10_list": hits10_list,
                     "mrr_list": mrr_list}

        else:
            y_pred = np.concatenate([y_pred_pos.reshape(-1,1), y_pred_neg], axis = 1)
            argsort = np.argsort(-y_pred, axis = 1)
            ranking_list = (argsort == 0).nonzero()
            ranking_list = ranking_list[1] + 1
            hits1_list = (ranking_list <= 1).astype(np.float32)
            hits3_list = (ranking_list <= 3).astype(np.float32)
            hits10_list = (ranking_list <= 10).astype(np.float32)
            mrr_list = 1./ranking_list.astype(np.float32)

            return {"hits@1_list": hits1_list, 
                     "hits@3_list": hits3_list,
                     "hits@10_list": hits10_list,
                     "mrr_list": mrr_list}


if __name__ == "__main__":
    ### hits case
    evaluator = Evaluator(name = "ogbl-ddi")
    print(evaluator.expected_input_format)
    print(evaluator.expected_output_format)
    # y_true = np.random.randint(2, size = (100,))
    y_pred_pos = torch.tensor(np.random.randn(100,))
    y_pred_neg = torch.tensor(np.random.randn(100,))
    input_dict = {"y_pred_pos": y_pred_pos, "y_pred_neg": y_pred_neg}
    result = evaluator.eval(input_dict)
    print(result)

    evaluator = Evaluator(name = "ogbl-wikikg")
    print(evaluator.expected_input_format)
    print(evaluator.expected_output_format)
    # y_true = np.random.randint(2, size = (100,))
    y_pred_pos = torch.tensor(np.random.randn(1000,))
    y_pred_neg = torch.tensor(np.random.randn(1000,100))
    input_dict = {"y_pred_pos": y_pred_pos, "y_pred_neg": y_pred_neg}
    result = evaluator.eval(input_dict)
    print(result['hits@1_list'].mean())
    print(result['hits@3_list'].mean())
    print(result['hits@10_list'].mean())
    print(result['mrr_list'].mean())

    y_pred_pos = y_pred_pos.numpy()
    y_pred_neg = y_pred_neg.numpy()
    input_dict = {"y_pred_pos": y_pred_pos, "y_pred_neg": y_pred_neg}
    result = evaluator.eval(input_dict)
    print(result['hits@1_list'].mean())
    print(result['hits@3_list'].mean())
    print(result['hits@10_list'].mean())
    print(result['mrr_list'].mean())


