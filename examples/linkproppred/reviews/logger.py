import torch


class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = torch.tensor(self.results[run])
            argmin = result[:, 1].argmin().item()
            print(f'Run {run + 1:02d}:')
            print(f'Lowest Train: {result[:, 0].min():.4f}')
            print(f'Lowest Valid: {result[:, 1].min():.4f}')
            print(f' Final Train: {result[argmin, 0]:.4f}')
            print(f'  Final Test: {result[argmin, 2]:.4f}')
        else:
            result = torch.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].min().item()
                valid = r[:, 1].min().item()
                train2 = r[r[:, 1].argmin(), 0].item()
                test = r[r[:, 1].argmin(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Lowest Train: {r.mean():.4f} ± {r.std():.4f}')
            r = best_result[:, 1]
            print(f'Lowest Valid: {r.mean():.4f} ± {r.std():.4f}')
            r = best_result[:, 2]
            print(f' Final Train: {r.mean():.4f} ± {r.std():.4f}')
            r = best_result[:, 3]
            print(f'  Final Test: {r.mean():.4f} ± {r.std():.4f}')

    def save(self, path):
        torch.save((self.info, torch.tensor(self.results)), path)

    @staticmethod
    def load(self, path):
        info, result = torch.load(path)
        logger = Logger(result.size(0), info)
        logger.results = result.tolist()
        return logger
