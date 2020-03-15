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

        self.task_type = meta_info[self.name]["task type"]

        if self.task_type == "link prediction":
            ### Hits@K
            self.K = 100


    def _parse_and_check_input(self, input_dict):
        if self.task_type == "link prediction":
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

        elif self.task_type == "link regression":
            if not "y_true" in input_dict:
                RuntimeError("Missing key of y_true")
            if not "y_pred" in input_dict:
                RuntimeError("Missing key of y_pred")

            y_true, y_pred = input_dict["y_true"], input_dict["y_pred"]

            """
                y_true: numpy ndarray or torch tensor of shape (num_edge, )
                y_pred: numpy ndarray or torch tensor of shape (num_edge, )
            """

            # converting to torch.Tensor to numpy on cpu
            if torch is not None and isinstance(y_true, torch.Tensor):
                y_true = y_true.detach().cpu().numpy()

            if torch is not None and isinstance(y_pred, torch.Tensor):
                y_pred = y_pred.detach().cpu().numpy()

            ## check type
            if not (isinstance(y_true, np.ndarray) and isinstance(y_true, np.ndarray)):
                raise RuntimeError("Arguments to Evaluator need to be numpy ndarray")

            if not y_true.shape == y_pred.shape:
                raise RuntimeError("Shape of y_true and y_pred must be the same")

            if not y_true.ndim == 1:
                raise RuntimeError("y_true and y_pred mush to 1-dim arrray, {}-dim array given".format(y_true.ndim))

            return y_true, y_pred

        else:
            raise ValueError("Undefined task type %s" (self.task_type))


    def eval(self, input_dict):

        if self.task_type == "link prediction":
            y_pred_pos, y_pred_neg, type_info = self._parse_and_check_input(input_dict)
            return self._eval_linkpred(y_pred_pos, y_pred_neg, type_info)
        elif self.task_type == "link regression":
            y_true, y_pred = self._parse_and_check_input(input_dict)
            return self._eval_linkregression(y_true, y_pred)
        else:
            raise ValueError("Undefined task type %s" (self.task_type))

    @property
    def expected_input_format(self):
        desc = "==== Expected input format of Evaluator for {}\n".format(self.name)
        if self.task_type == "link prediction":
            desc += "{\"y_pred_pos\": y_pred_pos, \"y_pred_pos\": y_pred_pos}\n"
            desc += "- y_pred_pos: numpy ndarray or torch tensor of shape (num_edge, )\n"
            desc += "- y_pred_neg: numpy ndarray or torch tensor of shape (num_edge, )\n"
            desc += "y_pred_pos is the predicted scores for positive edges.\n"
            desc += "y_pred_neg is the predicted scores for negative edges.\n"
            desc += "y_pred_neg needs to include the scores predicted on ALL the negative edges in the test or validation set.\n"
            desc += "Note: As the evaluation metric is ranking-based, the predicted scores need to be different for different edges."
        elif self.task_type == "link regression":
            desc += "{\"y_true\": y_true, \"y_pred\": y_pred}\n"
            desc += "- y_true: numpy ndarray or torch tensor of shape (num_edge, )\n"
            desc += "- y_pred: numpy ndarray or torch tensor of shape (num_edge, )\n"
            desc += "each row corresponds to one edge.\n"
            desc += "y_true is the target value to be predicted.\n"
            desc += "y_true should directly take valid_edge_label or test_edge_label as input.\n"
            desc += "y_pred should store the predicted target value.\n"
        else:
            raise ValueError("Undefined task type %s" (self.task_type))

        return desc

    @property
    def expected_output_format(self):
        desc = "==== Expected output format of Evaluator for {}\n".format(self.name)
        if self.task_type == "link prediction":
            desc += "{" + "hits@{}\": hits@{}".format(self.K, self.K) + "}\n"
            desc += "- hits@{} (float): Hits@{} score\n".format(self.K, self.K)
        elif self.task_type == "link regression":
            desc += "- rmse (float): root mean squared error"
        else:
            raise ValueError("Undefined task type %s" (self.task_type))

        return desc

    def _eval_linkpred(self, y_pred_pos, y_pred_neg, type_info):
        """
            compute Hits@K
        """

        if len(y_pred_neg) <= self.K:
            raise ValueError('Length of y_pred_neg is smaller than {}. Unable to evaluate Hits@{}.'.format(self.K, self.K))

        if type_info == 'torch':
            kth_score_in_negative_edges = torch.topk(y_pred_neg, self.K)[0][-1]
            hitsK = float(torch.sum(y_pred_pos > kth_score_in_negative_edges).cpu()) / len(y_pred_pos)

        # type_info is numpy
        else:
            kth_score_in_negative_edges = np.sort(y_pred_neg)[-self.K]
            hitsK = float(np.sum(y_pred_pos > kth_score_in_negative_edges)) / len(y_pred_pos)

        return {"hits@{}".format(self.K): hitsK}

    def _eval_linkregression(self, y_true, y_pred):
        """
            compute RMSE score
        """
        rmse = np.sqrt(((y_true - y_pred)**2).mean())

        return {"rmse": rmse}


if __name__ == "__main__":
    ### link prediction case
    evaluator = Evaluator(name = "ogbl-ppa")
    print(evaluator.expected_input_format)
    print(evaluator.expected_output_format)
    # y_true = np.random.randint(2, size = (100,))
    y_pred_pos = torch.tensor(np.random.randn(100,))
    y_pred_neg = torch.tensor(np.random.randn(1000,))
    input_dict = {"y_pred_pos": y_pred_pos, "y_pred_neg": y_pred_neg}
    result = evaluator.eval(input_dict)
    print(result)

    evaluator = Evaluator(name = "ogbl-reviews")
    print(evaluator.expected_input_format)
    print(evaluator.expected_output_format)
    y_true = np.random.randn(100,)
    y_pred = np.random.randn(100,)
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    result = evaluator.eval(input_dict)
    print(result)

