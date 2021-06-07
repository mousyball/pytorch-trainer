class AverageMeter:
    """A meter is for recording loss status.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self._avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value):
        self.value = value
        self.sum += value
        self.count += 1
        self._avg = self.sum / self.count

    @property
    def avg(self):
        return self._avg


class LossMeter:
    """A meter contains serveral meters for recording loss status.
    """

    def __init__(self):
        self.meters = None

    def init_meters(self, loss_key):
        """
        NOTE:
            loss_key is defined at network class
            set meters as a dictionary {key from loss_key: AverageMeter()}
        Args:
            loss_key (list): a list of key
        """
        meters = dict()
        for key in loss_key:
            meters[key] = AverageMeter()
        self.meters = meters

    def update(self, loss_tensors):
        """
        NOTE:
            * Update meters with values of loss_tensors
        Args:
            * loss_tensors (dict): a dictionary contains tensors of losses {key:scalr}
        """
        if self.meters is None:
            self.init_meters(list(loss_tensors.keys()))
        for key in loss_tensors.keys():
            if loss_tensors[key] is not None:
                self.meters[key].update(loss_tensors[key])

    def clear_meter(self):
        """Call the reset method in 'AverageMeter()' to clean up data"""
        for key in self.meters.keys():
            self.meters[key].reset()

    def clear(self):
        """clear all output in 'LossMeter()'"""
        self.meters = None

    def __repr__(self):
        # TODO: better format
        if self.meters is None:
            return 'LossMeter average:\n empty'
        string = [f' {key}:{val.avg}' for key, val in self.meters.items()]
        return 'LossMeter average:\n' + '\n'.join(string)
