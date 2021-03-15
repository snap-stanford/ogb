class BaseLoss(object):
    def __call__(self, score, label):
        pass

class BaseLogisticLoss(BaseLoss):
    """ Logistic Loss
    \log(1 + \exp(-l_i \cdot f(t_i)))
    l_i : label i from {-1, 1}
    f : score function
    t_i : triple i
    """
    def __init__(self):
        super(BaseLogisticLoss, self).__init__()

    def __call__(self, score, label):
        pass

class BaseBCELoss(BaseLoss):
    """ Binary Cross Entropy Loss
    -(l_i \cdot log(\sigma(f(t_i))) + (1 - l_i) \cdot \log(1 - \sigma(f(t_i))))
    l_i : label i from {0, 1}
    f : score function
    \sigma : logistic sigmoid function
    t_i : triple i
    """
    def __init__(self):
        super(BaseBCELoss, self).__init__()

    def __call__(self, score, label):
        pass

class BaseHingeLoss(BaseLoss):
    """ Hinge Loss
    \max(0, \lambda - l_i \cdot f(t_i))
    \lambda : margin value (hyper-parameter)
    l_i : label i
    f : score function
    t_i : triple i
    """
    def __init__(self, margin):
        super(BaseHingeLoss, self).__init__()
        self.margin = margin

    def __call__(self, score, label):
        pass

class BaseLogsigmoidLoss(BaseLoss):
    """ Logsigmoid Loss
    -\log(\frac{1}{1 + \exp(-l_i \cdot f(t_i))})
    l_i : label i from {-1, 1}
    f : score
    t_i : triple i
    """
    def __init__(self):
        super(BaseLogsigmoidLoss, self).__init__()

    def __call__(self, score, label):
        pass


class BaseLossGenerator(object):
    """ loss generator class is responsible for calculate loss for positive & negative loss / pairwise loss.
    It has different implementations of concrete method in regards of PyTorch and MXNet.
    """
    def __init__(self, neg_adversarial_sampling, adversarial_temperature, pairwise):
        """ initialize BaseLossGenerator class

        Parameters
        ----------
        neg_adversarial_sampling : bool
            whether to use adversarial sampling for negative sample
        adversarial_temperature : float
            temperature value for adversarial sampling
        pairwise : bool
            whether the loss computed is pairwise or point wise
        """
        self.pairwise = pairwise
        self.neg_adversarial_sampling = neg_adversarial_sampling
        if self.neg_adversarial_sampling:
            self.adversarial_temperature = adversarial_temperature
        else:
            self.adversarial_temperature = 0
        if self.pairwise is True and self.neg_adversarial_sampling is True:
            raise ValueError('loss cannot be pairwise and adversarial sampled')

    def get_pos_loss(self, pos_score, edge_weight):
        """ Predict loss for positive labels

        Parameters
        ----------
        pos_score : tensor
            Score calculated from positive triples
        edge_weight : tensor
            weight for each edge

        Returns
        -------
        tensor
            positive loss calculated with specific loss criterion
        """
        pass

    def get_neg_loss(self, neg_score, edge_weight):
        """ Predict loss for negative triples

        Parameters
        ----------
        neg_score: tensor
            Score calculated from positive triples
        edge_weight : tensor
            weight for each edge

        Returns
        -------
        tensor
            Negative loss calculated with specific loss criterion
        """
        pass

    def get_total_loss(self, pos_score, neg_score, edge_weight):
        """ Calculate total loss for a batch of positive triples and negative triples.
        The total loss can be point-wise and pairwise. For pairwise, it is average of the relative loss from positive score to negative
        score. For point-wise, it can be average of the positive loss and negative loss or negative loss
        weighted by its negative score and adversarial_temperature.

        If pairwise:
        \mathcal{L} = \frac{1}{|B|} \sum_{(t_i^+, t_i^-) \in B} L(f(t_i^-) - f(t_i^+)) \cdot w_{e_i}
        \mathcal{L} : total loss
        B : batch
        L : local loss criterion
        f : score function
        t_i^- : negative sample for triple i
        t_i^+ : positive sample for triple i
        w_{e_i} : weight for edge i

        If neg_adversarial_sampling:
        L_{adv\_neg} = \sum_{t_i^- \in B} softmax(f(t_i^-) \cdot T) \cdot L_{neg}
        B : batch
        L_{adv\_neg}-> adversarial weighed negative loss
        L_{neg} : negative loss
        f : score function
        t_i^- : negative sample for triple i
        T : adversarial temperature (hyper-parameter)

        Parameters
        ----------
        pos_score : tensor
            Score calculated from positive triples
        neg_score : tensor
            Score calculated from negative triples
        edge_weight : tensor
            weight for each edge

        Returns
        -------
        tensor
            Total loss by aggregate positive score and negative score.
        log
            log to record scalar value of negative loss, positive loss and/or total loss
        """
        pass