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

            if self.name == 'ogbl-citation':
                if not y_pred_pos.ndim == 1:
                    raise RuntimeError("y_pred_pos must to 1-dim arrray, {}-dim array given".format(y_pred_pos.ndim))

                if not y_pred_neg.ndim == 2:
                    raise RuntimeError("y_pred_neg must to 2-dim arrray, {}-dim array given".format(y_pred_neg.ndim))

                if not len(y_pred_pos) == len(y_pred_neg):
                    raise RuntimeError("Lengths of y_pred_pos and y_pred_neg need be the same.")

            else:
                if not y_pred_pos.ndim == 1:
                    raise RuntimeError("y_pred_pos must to 1-dim arrray, {}-dim array given".format(y_pred_pos.ndim))

                if not y_pred_neg.ndim == 1:
                    raise RuntimeError("y_pred_neg must to 1-dim arrray, {}-dim array given".format(y_pred_neg.ndim))

            return y_pred_pos, y_pred_neg, type_info

        elif 'mrr' == self.eval_metric:
            if 'scores' not in input_dict:
                raise RuntimeError("Missing key of scores")

            scores = input_dict['scores']

            if not isinstance(scores, torch.Tensor):
                raise ValueError("scores need to be torch.Tensor")

            if not scores.ndim == 2:
                raise RuntimeError("scores must be 2-dim array, {}-dim array given".format(scores.ndim))

            return scores

        else:
            raise ValueError("Undefined eval metric %s" % (self.eval_metric))


    def eval(self, input_dict):

        if "hits@" in self.eval_metric:
            y_pred_pos, y_pred_neg, type_info = self._parse_and_check_input(input_dict)
            if self.name == 'ogbl-citation':
                return self._eval_hits_citation(y_pred_pos, y_pred_neg, type_info)
            else:
                return self._eval_hits(y_pred_pos, y_pred_neg, type_info)
        elif self.eval_metric == 'mrr':
            if self.name == 'ogbl-wikikg':
                scores = self._parse_and_check_input(input_dict)
                return self._eval_mrr_wikikg(scores)
            else:
                raise ValueError("Undefined dataset %s" % self.name)
        else:
            raise ValueError("Undefined eval metric %s" % (self.eval_metric))

    @property
    def expected_input_format(self):
        desc = "==== Expected input format of Evaluator for {}\n".format(self.name)
        if "hits@" in self.eval_metric:
            if self.name == 'ogbl-citation':
                desc += "{\"y_pred_pos\": y_pred_pos, \"y_pred_neg\": y_pred_neg}\n"
                desc += "- y_pred_pos: numpy ndarray or torch tensor of shape (num_source_node, )\n"
                desc += "- y_pred_neg: numpy ndarray or torch tensor of shape (num_source_node, num_target_node_neg)\n"
                desc += "y_pred_pos[i] is a predicted score for a positive target node for the i-th source node.\n"
                desc += "y_pred_neg[i] is predicted scores for negative target nodes for the i-th source node.\n"
                desc += "Note: As the evaluation metric is ranking-based, the predicted scores need to be different for different edges."
            else:
                desc += "{\"y_pred_pos\": y_pred_pos, \"y_pred_neg\": y_pred_neg}\n"
                desc += "- y_pred_pos: numpy ndarray or torch tensor of shape (num_edge, )\n"
                desc += "- y_pred_neg: numpy ndarray or torch tensor of shape (num_edge, )\n"
                desc += "y_pred_pos is the predicted scores for positive edges.\n"
                desc += "y_pred_neg is the predicted scores for negative edges.\n"
                desc += "Note: As the evaluation metric is ranking-based, the predicted scores need to be different for different edges."
        elif self.eval_metric == 'mrr':
            if self.name == 'ogbl-wikikg':
                desc += "{\"scores\": scores, \"positive_args\": positive_args}\n"
                desc += "- scores: torch.Tensor of shape (batch_size, num_entities)\n"
                desc += "- positive_args: torch LongTensor of shape (batch_size, )\n"
                desc += "scores is the predicted scores for all the entities with filter bias added (check ogb/examples/linkproppred/wikikg/model.py).\n"
                desc += "positive_args is the index of the entity to evaluate.\n"
            else:
                raise ValueError("Undefined dataset %s" % self.name)
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
            if self.name == 'ogbl-wikikg':
                desc += "{" + "\"hits@1\": hits@1, \"hits@3\": hits@3, \n\"hits@10\": hits@10, \"mrr\": mrr}\n "
                desc += "- hits@1 (list of float): Hits@1 score\n" + \
                            "- hits@3 (list of float): Hits@3 score\n" + \
                            "- hits@10 (list of float): Hits@10 score\n" + \
                            "- mrr (list of float): MRR score\n"
            else:
                raise ValueError("Undefined dataset %s" % self.name)
        else:
            raise ValueError("Undefined eval metric %s" % (self.eval_metric))

        return desc

    def _eval_hits_citation(self, y_pred_pos, y_pred_neg, type_info):
        """
            compute Hits@K for ogbl-citation
            For each positive target node, the negative target nodes are different.

            y_pred_neg is a matrix
            rank y_pred_pos[i] agains y_pred_neg[i] for each i
        """

        if y_pred_neg.shape[1] < self.K:
            return {"hits@{}".format(self.K): 1.}

        if type_info == 'torch':
            kth_scores_in_negative_nodes = torch.topk(y_pred_neg, self.K, dim = 1)[0][:,-1]
            hitsK = float(torch.sum(y_pred_pos > kth_scores_in_negative_nodes).cpu()) / len(y_pred_pos)

        else:
            kth_scores_in_negative_nodes = np.sort(y_pred_neg, axis = 1)[:,-self.K]
            hitsK = float(np.sum(y_pred_pos > kth_scores_in_negative_nodes)) / len(y_pred_pos)

        return {"hits@{}".format(self.K): hitsK}


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
    
    def _eval_mrr_wikikg(self, scores):
        """
            compute mrr
            scores is an array with shape (batch size, num_entities).
            positive_args is an array with shape (batch size)
        """

        argsort = torch.argsort(scores, dim = 1, descending = True)
        # argsort = torch.transpose(torch.transpose(argsort, 0, 1) - positive_args, 0, 1)
        ranking = (argsort == 0).nonzero()
        ranking = ranking[:, 1] + 1
        hits1 = (ranking <= 1).to(torch.float).tolist()
        hits3 = (ranking <= 3).to(torch.float).tolist()
        hits10 = (ranking <= 10).to(torch.float).tolist()
        mrr = (1./ranking.to(torch.float)).tolist()

        return {"hits@1": hits1, 
                 "hits@3": hits3,
                 "hits@10": hits10,
                 "mrr": mrr}

if __name__ == "__main__":
    ### hits case
    evaluator = Evaluator(name = "ogbl-collab")
    print(evaluator.expected_input_format)
    print(evaluator.expected_output_format)
    # y_true = np.random.randint(2, size = (100,))
    y_pred_pos = torch.tensor(np.random.randn(100,))
    y_pred_neg = torch.tensor(np.random.randn(100,))
    input_dict = {"y_pred_pos": y_pred_pos, "y_pred_neg": y_pred_neg}
    result = evaluator.eval(input_dict)
    print(result)

    evaluator = Evaluator(name = "ogbl-citation")
    print(evaluator.expected_input_format)
    print(evaluator.expected_output_format)
    # y_true = np.random.randint(2, size = (100,))
    y_pred_pos = torch.tensor(np.random.randn(1000,))
    y_pred_neg = torch.tensor(np.random.randn(1000,10))
    input_dict = {"y_pred_pos": y_pred_pos, "y_pred_neg": y_pred_neg}
    result = evaluator.eval(input_dict)
    print(result)

