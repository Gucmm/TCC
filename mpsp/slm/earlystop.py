"""Critério de parada antecipada (puro, sem torch)."""


class EarlyStop:
    """Acompanha a loss de validação por época: guarda a melhor e manda parar após `patience` épocas sem melhora."""

    def __init__(self, patience: int = 3, min_delta: float = 0.0):
        self.patience, self.min_delta = patience, min_delta
        self.best, self.bad, self.best_epoch, self.history = float("inf"), 0, None, []

    def update(self, loss: float, epoch: float):
        """→ (melhorou, parar)."""
        self.history.append((epoch, loss))
        if loss < self.best - self.min_delta:
            self.best, self.bad, self.best_epoch = loss, 0, epoch
            return True, False
        self.bad += 1
        return False, self.bad >= self.patience
