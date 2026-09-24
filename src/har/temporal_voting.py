from collections import Counter, deque


class TemporalVoter:
    def __init__(self, window_size=5, minimum_votes=3):
        self.window_size = window_size
        self.minimum_votes = minimum_votes
        self.predictions = deque(maxlen=window_size)

    def update(self, prediction):
        self.predictions.append(prediction)

        counts = Counter(self.predictions)
        action, votes = counts.most_common(1)[0]

        if votes >= self.minimum_votes:
            return action

        return None