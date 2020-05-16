from sklearn.metrics import roc_auc_score
import pandas as pd
import os
import numpy as np

try:
    import torch
except ImportError:
    torch = None

### Evaluator for node property prediction
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

        self.num_tasks = int(meta_info[self.name]["num tasks"])
        self.eval_metric = meta_info[self.name]["eval metric"]


    def _parse_and_check_input(self, input_dict):
        if self.eval_metric == "rocauc" or self.eval_metric == "acc":
            if not "y_true" in input_dict:
                RuntimeError("Missing key of y_true")
            if not "y_pred" in input_dict:
                RuntimeError("Missing key of y_pred")

            y_true, y_pred = input_dict["y_true"], input_dict["y_pred"]

            """
                y_true: numpy ndarray or torch tensor of shape (num_node, num_tasks)
                y_pred: numpy ndarray or torch tensor of shape (num_node, num_tasks)
            """

            # converting to torch.Tensor to numpy on cpu
            if torch is not None and isinstance(y_true, torch.Tensor):
                y_true = y_true.detach().cpu().numpy()

            if torch is not None and isinstance(y_pred, torch.Tensor):
                y_pred = y_pred.detach().cpu().numpy()

            ## check type
            if not (isinstance(y_true, np.ndarray) and isinstance(y_true, np.ndarray)):
                raise RuntimeError("Arguments to Evaluator need to be either numpy ndarray or torch tensor")

            if not y_true.shape == y_pred.shape:
                raise RuntimeError("Shape of y_true and y_pred must be the same")

            if not y_true.ndim == 2:
                raise RuntimeError("y_true and y_pred must to 2-dim arrray, {}-dim array given".format(y_true.ndim))

            if not y_true.shape[1] == self.num_tasks:
                raise RuntimeError("Number of tasks for {} should be {} but {} given".format(self.name, self.num_tasks, y_true.shape[1]))

            return y_true, y_pred

        else:
            raise ValueError("Undefined eval metric %s " % (self.eval_metric))


    def eval(self, input_dict):

        if self.eval_metric == "rocauc":
            y_true, y_pred = self._parse_and_check_input(input_dict)
            return self._eval_rocauc(y_true, y_pred)
        elif self.eval_metric == "acc":
            y_true, y_pred = self._parse_and_check_input(input_dict)
            return self._eval_acc(y_true, y_pred)
        else:
            raise ValueError("Undefined eval metric %s " % (self.eval_metric))

    @property
    def expected_input_format(self):
        desc = "==== Expected input format of Evaluator for {}\n".format(self.name)
        if self.eval_metric == "rocauc":
            desc += "{\"y_true\": y_true, \"y_pred\": y_pred}\n"
            desc += "- y_true: numpy ndarray or torch tensor of shape (num_node, num_task)\n"
            desc += "- y_pred: numpy ndarray or torch tensor of shape (num_node, num_task)\n"
            desc += "where y_pred stores score values (for computing ROC-AUC),\n"
            desc += "num_task is {}, and ".format(self.num_tasks)
            desc += "each row corresponds to one node.\n"
        elif self.eval_metric == "acc":
            desc += "{\"y_true\": y_true, \"y_pred\": y_pred}\n"
            desc += "- y_true: numpy ndarray or torch tensor of shape (num_node, num_task)\n"
            desc += "- y_pred: numpy ndarray or torch tensor of shape (num_node, num_task)\n"
            desc += "where y_pred stores predicted class label (integer),\n"
            desc += "num_task is {}, and ".format(self.num_tasks)
            desc += "each row corresponds to one node.\n"
        else:
            raise ValueError("Undefined eval metric %s " % (self.eval_metric))

        return desc

    @property
    def expected_output_format(self):
        desc = "==== Expected output format of Evaluator for {}\n".format(self.name)
        if self.eval_metric == "rocauc":
            desc += "{\"rocauc\": rocauc}\n"
            desc += "- rocauc (float): ROC-AUC score averaged across {} task(s)\n".format(self.num_tasks)
        elif self.eval_metric == "acc":
            desc += "{\"acc\": acc}\n"
            desc += "- acc (float): Accuracy score averaged across {} task(s)\n".format(self.num_tasks)
        else:
            raise ValueError("Undefined eval metric %s " % (self.eval_metric))

        return desc

    def _eval_rocauc(self, y_true, y_pred):
        """
            compute ROC-AUC and AP score averaged across tasks
        """

        rocauc_list = []

        for i in range(y_true.shape[1]):
            #AUC is only defined when there is at least one positive data.
            if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == 0) > 0:
                is_labeled = y_true[:,i] == y_true[:,i]
                rocauc_list.append(roc_auc_score(y_true[is_labeled,i], y_pred[is_labeled,i]))

        if len(rocauc_list) == 0:
            raise RuntimeError("No positively labeled data available. Cannot compute ROC-AUC.")

        return {"rocauc": sum(rocauc_list)/len(rocauc_list)}

    def _eval_acc(self, y_true, y_pred):
        acc_list = []

        for i in range(y_true.shape[1]):
            is_labeled = y_true[:,i] == y_true[:,i]
            correct = y_true[is_labeled,i] == y_pred[is_labeled,i]
            acc_list.append(float(np.sum(correct))/len(correct))

        return {"acc": sum(acc_list)/len(acc_list)}

if __name__ == "__main__":
    ### rocauc case



    evaluator = Evaluator("ogbn-proteins")
    print(evaluator.expected_input_format)
    print(evaluator.expected_output_format)
    y_true = torch.tensor(np.random.randint(2, size = (100,112)))
    y_pred = torch.tensor(np.random.randn(100,112))
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    result = evaluator.eval(input_dict)
    print(result)

    ### acc case
    evaluator = Evaluator("ogbn-products")
    print(evaluator.expected_input_format)
    print(evaluator.expected_output_format)
    y_true = np.random.randint(5, size = (100,1))
    y_pred = np.random.randint(5, size = (100,1))
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    result = evaluator.eval(input_dict)
    print(result)

    ### acc case
    evaluator = Evaluator("ogbn-arxiv")
    print(evaluator.expected_input_format)
    print(evaluator.expected_output_format)
    y_true = np.random.randint(5, size = (100,1))
    y_pred = np.random.randint(5, size = (100,1))
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    result = evaluator.eval(input_dict)
    print(result)



